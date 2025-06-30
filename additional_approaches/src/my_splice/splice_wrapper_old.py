import torch
import splice
import numpy as np
from tqdm import tqdm
import math
import json 

class splice_wrapper():
    def __init__(self,model,device="cpu"):
        self.model = model
        self.vocab = np.array(splice.get_vocabulary("laion", 10000))
        self.concepts = torch.load(self.model.laion_embs_pth).to(device)
        self.device = device
        self.splicemodel = splice.SPLICE(image_mean = self.model.image_mean, dictionary = self.concepts, device=self.device, l1_penalty=0.25, return_weights=True, solver="skl")

    def force_compute_basis(self, embedding, rank, initial_l1 = None, return_attempts=False, MAX_ITERS=1000):
        '''
        This function iterativelly finds the optimal l1 penalty so that the l0 norm is exactly what indicated.
        If return attempts is true it saves other number of basis of other ranks than the one that is being searched for.
        An initial_l1 value can be passed as a hint to make the search faster.

        Embedding should have a batch dimension but be only one element. Shape: (1,n)
        '''
        #we'll use this to make smaller increments as we have more steps meaning harder to converge
       
        def step_weight(n_iters, value):

            weight = (- 1 / (1 + math.exp(-(0 + n_iters/200) - 3)))+ 1
            
            return value * weight
        
        if initial_l1 is None:
            l1p = 0.25
        else:
            l1p = initial_l1
        
        #In this dict we'll store the different l1 penalties and the results they've had
        results = {}

        l0n=None
        iter_cnt = 0

        while l0n != rank:
            iter_cnt += 1
            #Try with the current l1p
            #update the splice model with the new l1p
            self.splicemodel = splice.SPLICE(image_mean = self.model.image_mean, dictionary = self.concepts, device=self.device, l1_penalty=l1p, return_weights=True, solver="skl")
            
            weights, trunc_scores, words, l0n = self.get_main_concepts(embedding)
            
            #remove batchdim
            weights = weights[0,:]
            l0n = l0n[0]
            
            results[l0n] = {"l1p":l1p, "weights":weights,"trunc_scores": trunc_scores, "words":words}

            #If overshoot new l1p is half of itself more
            if l0n > rank:
                l1p = l1p + step_weight(iter_cnt, l1p/2)

            #If undershoot new l1p is half of itself less
            if l0n < rank:
                l1p = l1p - step_weight(iter_cnt, l1p/2)

            if iter_cnt > MAX_ITERS:
                print("Max iterations reach!")
                return None
        
        if return_attempts:
            return results
        
        else:
            return results[rank]

    def get_main_concepts(self,embedding, no_words=False, splice_model = None):
        '''
        This function only works in batch
        '''

        if splice_model == None:
            splice_model = self.splicemodel
        #first we need to center the embedding
        centered_emb = torch.nn.functional.normalize(embedding-splice_model.image_mean, dim=1)
        try:
            #compute the weights
            weights = splice_model.decompose(centered_emb)
        except:
            print("Something workng decomposingh!")
            print(centered_emb.shape)
            raise ValueError

        
        l0_norms = []
        for i in range(weights.shape[0]):
            l0_norms.append(int(torch.linalg.vector_norm(weights[i,:].squeeze(), ord=0).item()))

        sorted_idxs = torch.argsort(weights, descending= True, dim=1).detach().cpu()
        
        #we need the row indexes 
        row_indices = torch.arange(sorted_idxs.shape[0]).unsqueeze(1)
        row_indices = row_indices.expand(-1, sorted_idxs.shape[1])  

        if no_words:
            return weights, l0_norms
        
        else:
            words = []
            trunc_scores = []
            for i in range(weights.shape[0]):
                words.append(self.vocab[sorted_idxs[i,:l0_norms[i]].numpy()])
                trunc_scores.append(weights[i, sorted_idxs[i,:l0_norms[i]].numpy()])
            return weights, trunc_scores, words, l0_norms
    
    def reconstruct_emb(self, weights, splice_model=None):
        if splice_model == None:
            splice_model = self.splicemodel

        return splice_model.recompose_image(weights)

    def project_to_basis(self, weights, emb):
        '''
        We need this to be able to project the emb of a reconstructed image into the basis extracted from the hq.
        So we have a comparable filtered/simplified embedding
        '''

        def project(w, embedding):
            #First find the non_zero elements of the hq_img weights, these will be concepts we'll force to project our rec imag emb into
            non_zero_indices = torch.nonzero(w).ravel()

            #restrict concept vocabulary to this
            new_concepts = self.concepts[non_zero_indices,:]

            splicemodel = splice.SPLICE(image_mean = self.model.image_mean, dictionary = new_concepts, device=self.device, return_weights=True, l1_penalty=0.000000001) #No l0 penalty because we want it to use the whole concept dictionary here
            
            proj_weights, l0n = self.get_main_concepts(embedding.unsqueeze(0), no_words=True, splice_model=splicemodel)

            return proj_weights
        
        
        #check if batched or unbatched
        if emb.dim() == 2 and weights.dim()==2: #chek if batched
            #We have to treat batch in a loop beacuse different batches might have different basis rank
            reconstructions = []
            for b in range(weights.shape[0]):
                #First find the non_zero elements of the hq_img weights, these will be concepts we'll force to project our rec imag emb into
                non_zero_indices = torch.nonzero(weights[b,:]).ravel()

                #restrict concept vocabulary to this
                new_concepts = self.concepts[non_zero_indices,:]

                splicemodel = splice.SPLICE(image_mean = self.model.image_mean, dictionary = new_concepts, device=self.device, return_weights=True, l1_penalty=0.0001) #No l0 penalty because we want it to use the whole concept dictionary here
                
                proj_weights, l0n = self.get_main_concepts(emb[b,:].unsqueeze(0), no_words=True, splice_model=splicemodel)
                reconstructions.append(self.reconstruct_emb(proj_weights,splicemodel).squeeze(0))  

            return torch.stack(reconstructions)
        
        #unbatched
        elif emb.dim()==1 and weights.dim()==1:
            return project(weights,emb)
        else:
            print("Got different batching!")
            print(emb.shape)
            print(weights.shape)
            raise ValueError
            


    def find_all_basis_rank_range(self, embedding, start_rank, end_rank, l1p_hint_pth=None):
        '''
        Find all basis in a given range of ranks.
        '''

        #Check if a precomputed l1p hints has been computed for this model
        if l1p_hint_pth is not None:
            with open(l1p_hint_pth) as json_file:
                l1p_hint  = json.load(json_file)
                #change keys to int and values to float
                l1p_hint = {int(k):float(v) for k,v in l1p_hint.items()}

        else:
            l1p_hint = None


        #Start with the rank in the middle, then iterativelly search from less to more
        middle_rank = int((end_rank - start_rank)/2 + start_rank)
        data = self.force_compute_basis(embedding, middle_rank, return_attempts=True)

        #now iterativelly search from bottom to top
        for rank in tqdm(range(start_rank,end_rank+1)):
            known_ranks = np.array(list(data.keys()))
            if rank in known_ranks:
                continue #skip iteration
            else:
                aprox_flag = False
                #If we have l1p_hint
                if l1p_hint is not None:
                    if rank in l1p_hint.keys():
                        l1p = l1p_hint[rank]
                    else:
                        print("not known")
                        aprox_flag = True
                else:
                    ("print no l1phint")
                    aprox_flag = True

                #If we don't we approximate initial l1p penalty with the two closest ranks
                if aprox_flag:

                    #look for closest rank
                    diffs = known_ranks - rank
                    closest_idxs = np.argsort(diffs)
                    
                    close_1_r = known_ranks[closest_idxs[0]]
                    close_2_r = known_ranks[closest_idxs[1]]

                    closest_l1p = data[close_1_r]["l1p"]
                    second_l1p=  data[close_2_r]["l1p"]

                    l1penalties = np.array([closest_l1p,second_l1p])
                    #now compute optimal initial l1p
                    l1p = float(np.min(l1penalties) + np.abs(closest_l1p - second_l1p))
                
                #now compute that rank
                data.update(self.force_compute_basis(embedding, rank, initial_l1=l1p, return_attempts=True))
        
        return data
    
    def save_data_as_l1p_hint(self, data:dict, path:str):
        '''
        Makes a dictionaru with rank:l1p and saves it as a json
        '''
        my_data = {}
        #Remove from data the embeddings
        for rank in data.keys():
            my_data[rank] = data[rank]["l1p"]
        
        #save as a json 
        with open(path, "w") as outfile: 
            json.dump(my_data, outfile)