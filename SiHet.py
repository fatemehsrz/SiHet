import torch
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
from loss import LossNegSampling
import time

USE_CUDA = False # torch.cuda.is_available()
#gpus = [0]
#torch.cuda.set_device(gpus[0])
    
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
        
class SiHet():
    
    def __init__(self,  name='wiki' , emb_file=None,  emb_size= 2, alpha=5, epoch=5, neg_samples=5, batch_size= 256, shuffel=True):

        self.emb_size = emb_size
        self.shuffel = shuffel
        self.emb_file= emb_file
        self.neg_samples = neg_samples
        self.batch_size=batch_size
        self.epoch=epoch
        self.alpha= alpha
        self.name=name
        self.index2word = dict()
        self.word2index = dict()
        self.read_data(self.name)
        self.build_vocab()

    def read_data(self, name):

        self.sentiments = []
        self.social_edges=[]

        f_sent= open('./data/%s_sentiment.txt'%name, 'r')
        f_social= open('./data/%s_social.txt'%name, 'r')

        sent_counter=0
        social_counter=0

        for line in f_sent:
            a=line.strip('\n').split(' ')
            self.sentiments.append((a[0],a[1], a[2]))
            sent_counter+=1

        for line in f_social:
            a=line.strip('\n').split(' ')
            self.social_edges.append((a[0],a[1]))
            social_counter+=1

        print(self.name, 'number of social links:', social_counter, 'number of sentiment links:', sent_counter )

    def getBatch(self, batch_size, train_data):
        
        if self.shuffel==True:
            random.shuffle(train_data)
        
        sindex = 0
        eindex = batch_size
        while eindex < len(train_data):
            batch = train_data[sindex: eindex]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch
        
        if eindex >= len(train_data):
            batch = train_data[sindex:]
            
            yield batch
            
            
    def prepare_sequence(self, seq, word2index):
        idxs = list(map(lambda w: word2index[w], seq))
        return Variable(LongTensor(idxs))
    
    def prepare_word(self,word, word2index):
        return Variable(LongTensor([word2index[str(word)]]))
    
    def prepare_weight(self, weight):
        return  FloatTensor([weight])
     
    
    def build_vocab(self):
        
        self.sent_nodes=[]
        self.social_nodes=[]
        
        for u,v, sent in self.sentiments:
            self.sent_nodes.append(u)
            self.sent_nodes.append(v)
            
            
        for u,v in self.social_edges:
            self.social_nodes.append(u)
            self.social_nodes.append(v)   
            
            
        self.all_nodes= set(self.sent_nodes + self.social_nodes) 
        self.num_nodes=len(self.all_nodes)

        self.word2index = {}
        for vo in self.all_nodes:
            if self.word2index.get(vo) is None:
                self.word2index[str(vo)] = len(self.word2index)
                

        self.index2word = {v:k for k, v in self.word2index.items()}
        
        
               
    def prepare_trainData(self):    
       
        print('prepare training data ...')
        
        self.train_data = []
        social={}
        
        self.neg_list=[]
        self.pos_list=[]

        for u_social, v_social  in self.social_edges:
            
            social[u_social]= v_social
            social[v_social]= u_social
         
        

        for u_sent, v_sent, sent  in self.sentiments:
            
            if sent== '1':
                    
                v_scl= social[u_sent]

                print('pos link +1:' , (u_sent, v_sent), 'social:',(u_sent,v_scl))
                
                self.pos_list.append(v_sent)
             
                for i in range(self.alpha): 
                    
                   self.train_data.append((u_sent, v_sent, v_scl ))
                   

            else:
                
                self.neg_list.append(v_sent)
                print('neg link -1:', (u_sent,v_sent))


        #self.neg_list= list( set(self.sent_nodes)-set(self.pos_list))
        

        u_p = []
        v_sent_p = []
        v_social_p = []

        tr_num=0    
        
        for tr in self.train_data:

            u_p.append(self.prepare_word(tr[0], self.word2index).view(1, -1))
            v_sent_p.append(self.prepare_word(tr[1], self.word2index).view(1, -1))
            v_social_p.append(self.prepare_word(tr[2], self.word2index).view(1, -1))
            
            tr_num+=1

          
        train_samples = list(zip(u_p, v_sent_p, v_social_p))
        
        print(len(train_samples), 'samples are ready ...')
        
        return train_samples
        

    def negative_sampling(self, targets_sent,  k):
            
        batch_size = targets_sent.size(0)
        
        
        neg_samples = []
         
        for i in range(batch_size):
             
            nsample = []
            target_index = targets_sent[i].data.cpu().tolist()[0] if USE_CUDA else targets_sent[i].data.tolist()[0]
            v_node= self.index2word[target_index]
            
            while len(nsample) < k: # num of sampling
                
                neg = random.choice(self.neg_list)
                nsample.append(neg)

               
            neg_samples.append(self.prepare_sequence(nsample, self.word2index).view(1, -1))
        
        return torch.cat(neg_samples)        

   

    def train (self):

        train_data= self.prepare_trainData()
        
        final_losses = []

        model = LossNegSampling(self.num_nodes, self.emb_size)
        
        if USE_CUDA:
           model = model.cuda()
           
        optimizer = optim.Adam(model.parameters(), lr=0.001)
       
        self.epoches=[]
        
        #f_loss=open('./loss/%s_time_size_%d_alpha_%d.txt'%(self.name, self.emb_size, self.alpha), 'w')
        
        for epoch in range(self.epoch):
            
            t1=time.time() 
            
            for i,  batch in enumerate(self.getBatch(self.batch_size, train_data)):

                inputs_sent, targets_sent,targets_social = zip(*batch)
               
                inputs_sent = torch.cat(inputs_sent) # B x 1
                targets_sent=torch.cat( targets_sent) # B x 1
                targets_social=torch.cat( targets_social) # B x 1

                negs = self.negative_sampling(targets_sent , self.neg_samples)
    
                model.zero_grad()
                
                final_loss = model(inputs_sent, targets_sent, targets_social,  negs)
                
                final_loss.backward()
                optimizer.step()
            
                final_losses.append(final_loss.data.cpu().numpy())
               
          
            t2= time.time()
            print(self.name, 'loss: %0.3f '%np.mean(final_losses),  'Epoch time: ', '%0.4f'%(t2-t1), 'dimension size:', self.emb_size,' alpha: ', self.alpha )
                
            #f_loss.write(str('final loss: %0.3f '%np.mean(final_losses) ) +' samples_num: '+str(len(self.train_data))+ str(' epoch time: %0.3f '%(t2-t1) )+
                         #' emb_size: '+str(self.emb_size)+ ' alpha: '+str(self.alpha))
            #f_loss.write('\n')
          
        
        #f_loss.close()
             
               
        normal_emb={}

        for w in self.sent_nodes:
            vec=model.get_emb(self.prepare_word(w, self.word2index))
            normal_emb[w]=vec.data.cpu().numpy()[0]

            
        return normal_emb
                  
            
    
    
           

        

        
