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

    
    def _decompose_rank(self, 
                        embedding:torch.tensor, 
                        method:str, 
                        rank:int, 
                        return_attempts:bool=False
                        ):
        '''
        This function iterativelly finds the optimal l1 penalty so that the l0 norm is exactly what indicated.
        If return attempts is true it saves other number of basis of other ranks than the one that is being searched for.
        An initial_l1 value can be passed as a hint to make the search faster.

        Embedding should have a batch dimension.

        embedding : tensor to decompose
        method: skl or admm (skl breaks the computation graph)
        rank,
        return_attempts: if true willl return a dict with all the attempts

        Binary search
        '''
        
        def aproximate_l1p(rank:int):
            '''
            Given a rank, looks at l1p_hints and finds the upper and lower l1p known values
            Note that some ranks might not be found and will return None
            '''
            #define left and right margins
            #get known ranks 
            Ranks = np.array(list(self.l1p_hint[method].keys()))
            
            '''
            Because rank and l1p have an inverse relation,
            The lower l1p bound should be the highest lowest upper rank we know
            And the higher l1p bound should be the highest under rank we know
            '''
            #get ranks bigger than target
            higher_indxs = np.where(Ranks > rank)[0]
            if len(higher_indxs) > 0:
                upper_rank = Ranks[higher_indxs].min()
            else:
                upper_rank =  None

            lower_indxs = np.where(Ranks < rank)[0]
            if len(lower_indxs) > 0:
                lower_rank = Ranks[lower_indxs].max()
            else:
                lower_rank =  None


            #convert upper_bound from rank to l1p value
            #check again if the rank is known, otherwise default
            if upper_rank is not None:
                lower_bound = self.l1p_hint[method][upper_rank]
            else:
                lower_bound = 0.01
            
            if lower_rank is not None:
                upper_bound = self.l1p_hint[method][lower_rank]
            else:
                upper_bound = 1
                
            #apply a small margin
            #upper_bound = upper_bound * 0.8 
            #lower_bound = lower_bound * 1.20
            
            #computed weighted average by their rank distances to the one we are looking for
            #check if we have both bounds, otherwise simple average
            '''
            if lower_rank is None or upper_rank is None:
                diff_up_b = 0.5
                diff_low_b = 0.5
            
            else:
                diff_up_b = abs(rank - lower_rank)
                diff_low_b = abs(rank - upper_rank)
                diff_sum = diff_up_b + diff_low_b
                
                diff_up_b /= diff_sum
                diff_low_b /= diff_sum
            
            l1p = upper_bound *  diff_up_b +  lower_bound * diff_low_b
            '''
            l1p = (upper_bound + lower_bound) / 2
            return l1p
        
        results_batch = []
        weights_batch = []

        #iterate batches
        for bidx in range(embedding.shape[0]):
    
            #check if we have already computed decompostion for this same rank
            if rank in self.l1p_hint[method].keys():
                l1p = self.l1p_hint[method][rank]
            
            #otherwise take middle of bounds
            else:
                #make it weighted average by distance from the
                l1p = aproximate_l1p(rank)
            

            #In this dict we'll store the different l1 penalties and the results they've had
            results = {}

            l0n=None
            iter_cnt = 0

            while l0n != rank:
                iter_cnt += 1
                #Try with the current l1p
                weights = self._decompose_l1(embedding[bidx,:].unsqueeze(0), method, l1p)

                #compute l0norm
                l0n = int(self.weights2l0n(weights)[0].item())

                #save to results
                results[l0n] = weights

                #save l1phint
                #replace with new value
                self.l1p_hint[method][l0n] =  float(l1p)

                #if didn't hit target rank recompute bounds, every iter l1p_hint will update 
                l1p = aproximate_l1p(rank)

                if iter_cnt > self.rank_search_max_iter:
                    print("Max iterations reached searching for rank! returning None")
                    return None
                
            #save weights
            weights_batch.append(weights.squeeze(0))

            #save results
            results_batch.append(results)
        
        weights_batch = torch.stack(weights_batch)

        if return_attempts:
            return results_batch
            
        else:
            return weights_batch
        

    def _decompose_l1(
            self,
            embedding:torch.tensor,
            solver : str,
            l1:float = None,
            ) -> torch.tensor:
        '''
        Decomposes the given model embedding vector into a set of sparce weights for the given vocabulary
        
        embedding: embedding to actually decompose (should come batched)
        l1 : l1 penalty when decomposing, (non compatible with rank)
        solvers : l1
        '''

        if solver == 'skl':
            l1p_skl = l1/(2*self.image_mean.shape[0])  ## skl regularization is off by a factor of 2 times the dimensionality of the CLIP embedding. See SKL docs.
            clf = linear_model.Lasso(alpha=l1p_skl, fit_intercept=False, positive=True, max_iter=10000, tol=1e-6)
            skl_weights = []
            if embedding.requires_grad: 
                print("Warning: The passed embedding has gradients, the output using the skl solver does not. We are breaking the gradient computation graph here.")
                embedding = embedding.clone().detach()

            for i in range(embedding.shape[0]): #iter batch
                clf.fit(self.concepts.T.cpu().numpy(), embedding[i,:].cpu().numpy())
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
        l1:float = None,
        rank:int = None,
        return_attempts:bool = False

        ) -> torch.tensor:
        '''
        Decomposes the given model embedding vector into a set of sparce weights for the given vocabulary
        
        embedding: embedding to actually decompose (should come batched)
        l1 : l1 penalty when decomposing, (non compatible with rank)
        rank : rank of sparse vector to decompose into (non compatible with l1)
        solvers : l1
        return_attempts : only available for rank decomposition. Returns results in different attempts when finding the rank decomposition.
        '''

        assert l1 is not None or rank is not None, "Should provide at least l1 or rank!!"

        assert not (l1 is not None and rank is not None), "l1 and rank are non compatible, only provide one"
        
        #normalize embedding
        emb = torch.nn.functional.normalize(embedding, dim=1)
        
        #center emb
        emb = torch.nn.functional.normalize(embedding-self.image_mean, dim=1)

        if l1 is not None:
            return self._decompose_l1(emb, solver, l1)
        
        else:
            return self._decompose_rank(emb, solver, rank, return_attempts)
    
    
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

    def recompose(self,
                  weights:torch.tensor
                  ) -> torch.tensor:
        '''
        Recomposes a weights tensor into an embedding again by doing weighted sum of concept dictionary embeddings
        '''
        recons = weights@self.concepts
        recons = torch.nn.functional.normalize(recons, dim=1)
        recons = torch.nn.functional.normalize(recons+ self.image_mean, dim=1)
        return recons