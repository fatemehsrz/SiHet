
from SiHet import SiHet
import sign_link_pred as sp
import node_recommend as nc


if __name__=='__main__':
    
    for name in [ 'Weibo' ]: # 'Weibo' ,'wiki'

        vec_model= SiHet(  name,  emb_size= 100, alpha= 5, epoch=5, neg_samples=5, batch_size=256, shuffel=True)
                
        embeddings= vec_model.train()
        sp.link_pred(name, embeddings)
        nc.node_recom(name, embeddings)
                 
          
            
