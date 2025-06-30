import torch
import numpy as np
import json

class greedy_decomposer():
    def __init__(self, 
                 concepts_pth:torch.Tensor,
                 vocab:np.array = None):
        ''' 
        Greedy decomposition module. 
        Change self.rank to change target rank default:10
        mode: debug, w_filt_rec, filt_rec, w_rec, rec

        debug: ( filtered concepts_embeddings, text) tuple
        w_filt_rec: (weighted sum using cosine of filterec concept embeddings)
        filt_rec: (average of filtered concept embeddings)
        w_rec: (weighted sum using cosine of concept embeddings)
        rec: average of concept embeddings.
        '''
        #vocab
        if vocab is None:
            with open("src/my_splice/laion1000_vocab.json", 'r') as f:
                self.vocab = np.array(json.load(f))

        else:
            self.vocab = vocab

        #load concepts from pth
        self.Concepts = torch.load(concepts_pth)
        
        self.rank=50
        self.mode = "debug" #by default returns stack of embeddings(filtered)


    def __call__(self,
                 target:torch.Tensor):
        
        '''
        Implementation of the text span algorithm
        greedy decomposition
        C: concepts embeddings
        R: target vector
        rank: rank of decomposition
        '''

        meta_target = target.clone()
        meta_concepts = self.Concepts.clone()

        C = []
        C_ = []
        texts = []
        for i in range(self.rank):
            D = torch.matmul(meta_target,meta_concepts.T) #dot products
            j = torch.argmax(D) #find max dot product

            #add concept text
            texts.append(self.vocab[j]) 
            
            #concept_emb
            concept_emb = meta_concepts[j,:]
            n_concept_emb = concept_emb / torch.linalg.norm(concept_emb)

            #save component
            C_.append(concept_emb)#save concept embedding
            C.append(self.Concepts[j,:])

            #project target onto concept to remove redundant feature
            p_t = n_concept_emb * torch.dot(n_concept_emb, meta_target)
            meta_target -= p_t

            #remove found concept component also from concept dictionary
            dots = torch.matmul(n_concept_emb, meta_concepts.T)
            meta_concepts -= n_concept_emb.repeat(len(dots),1) * dots[:,None]
        
        C_ =  torch.stack(C_)
        C = torch.stack(C)

        if self.mode == "debug":
            return C_, C, texts
    
        if self.mode == "w_filt_rec":
            #compute cosine similarities between filtered concepts and target
            cosines = torch.nn.functional.cosine_similarity(target, C_)
            
            #normalize cosines so they add up to one
            cosines /= cosines.sum()

            cosines =  cosines.unsqueeze(1).expand(C_.shape[0],C_.shape[1])
            weighted = C_ * cosines
            return torch.sum(weighted,dim=0)
        
        if self.mode == "filt_rec":
            return torch.mean(C_,dim=0)

        if self.mode == "w_rec":
            #compute cosine similarities between filtered concepts and target
            cosines = torch.nn.functional.cosine_similarity(target, C)
            
            #normalize cosines so they add up to one
            cosines /= cosines.sum()

            cosines =  cosines.unsqueeze(1).expand(C.shape[0],C.shape[1])
            weighted = C * cosines
            return torch.sum(weighted,dim=0)
        
        if self.mode == "rec":
            return torch.mean(C,dim=0)
