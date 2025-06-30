import numpy as np
import math
import json 
import torch
from sklearn import linear_model
from src.my_splice.admm import ADMM

class splice_wrapper():
    def __init__(
            self,
            concepts_pth:str, 
            model_img_mean_pth:str, 
            device:str, 
            l1_hints_pth:str = None, 
            vocab:np.array=None,
            rank_search_max_iter:int=1000
            ):
        '''
        concepts_pth : pth to torch tensor pt file with the embeddings of the vocab words
        model_emb_mean_pth : pth to torch tensor pt file with embedding mean for the given model
        device : torch device 
        l1_hints_pth : json file pth with the l1 value hints for different ranks 
        vocab : np array with all the vocabulary words as strings for the concepts, if none use laion 
        rank_search_max_iter : maximum number of iterations when finding l1 value for a specific rank
        '''

        #device
        self.device = device

        #max iters
        self.rank_search_max_iter = rank_search_max_iter
        #vocab
        if vocab is None:
            with open("src/my_splice/laion1000_vocab.json", 'r') as f:
                self.vocab = np.array(json.load(f))

        else:
            self.vocab = vocab

        #concepts
        self.concepts = torch.load(concepts_pth).to(device)

        #emb mean
        self.image_mean =  torch.load(model_img_mean_pth).to(device)
        #l1hints
        if l1_hints_pth is not None:
            with open(l1_hints_pth) as json_file:
                l1p_hint  = json.load(json_file)
                #change keys to int and values to float
                self.l1p_hint = {int(k):float(v) for k,v in l1p_hint.items()}

        else:
            self.l1p_hint = {"skl":{},"admm":{}} #otherwise to empty dict to stat to fill in
    
    def savel1p_hint(self,
                     pth:str
                     ):
        '''
        Saves current splice module l1p hint to a json file
        '''
        #save as a json 
        with open(pth, "w") as outfile: 
            json.dump(self.l1p_hint, outfile)

    def _decompose_l1(
            self,
            embedding:torch.tensor,
            solver : str,
            l1:float = None,
            concepts: torch.tensor = None
            ) -> torch.tensor:
        '''
        Decomposes the given model embedding vector into a set of sparce weights for the given vocabulary
        
        embedding: embedding to actually decompose (should come batched)
        l1 : l1 penalty when decomposing, (non compatible with rank)
        solvers : "skl" or "admm"
        concepts : if not passed decomposition on default init concept space
        '''

        if solver == 'skl':
            l1p_skl = l1/(2*self.image_mean.shape[0])  ## skl regularization is off by a factor of 2 times the dimensionality of the CLIP embedding. See SKL docs.
            clf = linear_model.Lasso(alpha=l1p_skl, fit_intercept=False, positive=True, max_iter=10000, tol=1e-6)
            skl_weights = []
            if embedding.requires_grad: 
                print("Warning: The passed embedding has gradients, the output using the skl solver does not. We are breaking the gradient computation graph here.")
                embedding = embedding.clone().detach()

            for i in range(embedding.shape[0]): #iter batch
                if concepts is None:
                    clf.fit(self.concepts.T.cpu().numpy(), embedding[i,:].cpu().numpy())
                else:
                    clf.fit(concepts.T.cpu().numpy(), embedding[i,:].cpu().numpy())
                
                skl_weights.append(torch.tensor(clf.coef_))
            
            weights = torch.stack(skl_weights, dim=0).to(self.device)
        
        elif solver == 'admm':
            admm = ADMM(rho=5, l1_penalty=l1, tol=1e-6, max_iter=2000, device=self.device, verbose=False)
            weights = admm.fit(self.concepts, embedding).to(self.device)
        
        #return result
        return weights
        
    def decompose(
        self,
        embedding:torch.tensor,
        solver : str,
        l1:float,
        concepts:torch.tensor = None

        ) -> torch.tensor:
        '''
        Decomposes the given model embedding vector into a set of sparce weights for the given vocabulary
        
        embedding: embedding to actually decompose (should come batched)
        l1 : l1 penalty when decomposing, (non compatible with rank)
        solvers : "skl" or "admm"
        concepts : if not passed decomposition on default init concept space

        '''

        #normalize embedding
        emb = torch.nn.functional.normalize(embedding, dim=1)
        
        #center emb
        emb = torch.nn.functional.normalize(embedding-self.image_mean, dim=1)

        if l1 is not None:
            return self._decompose_l1(emb, solver, l1, concepts)
   
    def recompose(self,
                  weights:torch.tensor,
                  concepts:torch.tensor = None
                  ) -> torch.tensor:
        '''
        Recomposes a weights tensor into an embedding again by doing weighted sum of concept dictionary embeddings
        '''
        if concepts is None:
            recons = weights@self.concepts
        else:
            recons = weights@concepts
            
        recons = torch.nn.functional.normalize(recons, dim=1)
        recons = torch.nn.functional.normalize(recons+ self.image_mean, dim=1)
        return recons
    
    def project2otherweights(self,
                             embedding:torch.tensor,
                             weights:torch.tensor,
                             l1:torch.tensor,
                             solver:str = "skl"
                             ):
        '''
        Given an embedding an another embedding decomposition weights on main concepts vocabulary (the one class inititialized)
        project the embedding into the concept space represented in the weights
        '''

        
        indexes = torch.where(weights > 0)[1] #non-neagtive weights
        new_concepts = self.concepts[indexes,:]

        dec_weights = self.decompose(embedding,solver,l1,new_concepts)
        recomp = self.recompose(dec_weights, new_concepts)
        
        #we should return weights of the same shape as the original 
        out_weights = torch.zeros_like(weights)

        #put in the computes values at the corresponding index
        for j,i in enumerate(indexes):
            out_weights[0,i] = dec_weights[0,j]


        return recomp, out_weights
    
    def weights2l0n(self, 
                    weights:torch.tensor
                    ) -> torch.tensor:
        '''
        Computes the l0 norm of the weights tensor passed
        '''
        return torch.linalg.vector_norm(weights, ord=0, dim=1)

    def weights2words(
            self,
            weights:torch.tensor
            ) -> tuple[ torch.tensor , list[str], torch.tensor]:
        '''
        Given a weights tensor returns the l0 norms for the weights tesnors, 
        the decomposed words and the respective scores
        '''
        words = []
        trunc_scores = []

        sorted_idxs = torch.argsort(weights, descending= True, dim=1).detach().cpu()
        l0_norms = self.weights2l0n(weights)

        #iterate batch
        for i in range(weights.shape[0]):
            words.append(self.vocab[sorted_idxs[i,:int(l0_norms[i])].numpy()])
            trunc_scores.append(weights[i, sorted_idxs[i,:int(l0_norms[i])].numpy()])
        return l0_norms, words, trunc_scores

    