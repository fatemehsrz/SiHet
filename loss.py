import torch
import torch.nn as nn



class LossNegSampling(nn.Module):
    
    def __init__(self, vocab_size, emb_dim):
        super(LossNegSampling, self).__init__()
        
        self.embedding_u = nn.Embedding(vocab_size, emb_dim) #  embedding  u
        
        self.logsigmoid = nn.LogSigmoid()
                
        initrange = (2.0 / (vocab_size + emb_dim))**0.5 #  init
        self.embedding_u.weight.data.uniform_(-initrange, initrange) # init u
        
        
       
    def forward(self, u_sent, v_sent, v_social, negative_nodes_sent):
        
        u_embed_sent = self.embedding_u(u_sent) # B x 1 x Dim  edge (u,v)
        
        v_embed_sent = self.embedding_u(v_sent) # B x 1 x Dim  
        v_embed_social = self.embedding_u(v_social) # B x 1 x Dim  
       
        negs_sent = -self.embedding_u(negative_nodes_sent) # B x K x Dim  neg samples
        
        positive_score_sent=  v_embed_sent.bmm(u_embed_sent.transpose(1, 2)).squeeze(2) # Bx1
        positive_score_social=  v_embed_social.bmm(u_embed_sent.transpose(1, 2)).squeeze(2) # Bx1
  
        negative_score_sent= torch.sum(negs_sent.bmm(u_embed_sent.transpose(1, 2)).squeeze(2), 1).view(negative_nodes_sent.size(0), -1) # BxK -> Bx1
         
        sum_all = self.logsigmoid(positive_score_sent)+self.logsigmoid(positive_score_social) + self.logsigmoid(negative_score_sent)
        
        loss= -torch.mean(sum_all)

        return loss
    
    
    def get_emb(self, inputs):
        embeds = self.embedding_u(inputs) ### u
        

        return embeds



    
   
