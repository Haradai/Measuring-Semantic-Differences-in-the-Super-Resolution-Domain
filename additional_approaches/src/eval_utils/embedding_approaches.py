from sklearn.cluster import KMeans
import torch
import numpy as np
from src.eval_utils.compute_metrics import compute_embedding_sim_metrics

def cluster_concepts(model, concepts:list[str]) -> list[str]:
    ''' 
    Using model's text embedder cluster concepts into the main 4 concepts
    '''
    def kmeans_clustering(vectors,n_clusters=3):        
        # Convert PyTorch tensor to NumPy array
        vectors_np = vectors.detach().cpu().numpy()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors_np)
        
        # Convert cluster labels back to PyTorch tensor
        cluster_labels_tensor = torch.from_numpy(cluster_labels)
        
        return cluster_labels_tensor
    
    #check if number of concepts <= number clusters
    if len(concepts) <= 3:
        return concepts
    #embed text
    embs = model.txt_embedder(concepts)
    
    main_words_idx = np.unique(kmeans_clustering(embs))

    concepts = np.array(concepts)[main_words_idx].tolist()
    return concepts


def simple_embedding_similarity(model, ram_model, metrics, target_layer , hq_pth:str, rec_pth:str, data:list,do_gradcam=True):
    
    for metric in metrics:
        res = compute_embedding_sim_metrics(model.img_embedder ,target_layer ,hq_pth ,rec_pth, metric, do_gradcam=do_gradcam)
        
        #add approach name
        res["embedding_method"] = "simple" 
        #store result
        data.append(res)

def focus_embedding_similarity(model,ram_model, metrics, target_layer, hq_pth:str, rec_pth:str, data:list,do_gradcam=True):

    #generate captions using ram model 
    img_elements = ram_model(hq_pth)
    img_elements = cluster_concepts(model, img_elements)
    for elem in img_elements: #generate new row per word
        focus_emb = model.txt_embedder(elem)
        #set the focus embedding
        model.focus_img_embedder.focus_emb = focus_emb

        for metric in metrics:
            res = compute_embedding_sim_metrics(model.focus_img_embedder ,target_layer ,hq_pth ,rec_pth, metric,do_gradcam=do_gradcam)
            
            #add text element
            res["img_element"] = elem
            #add approach name
            res["embedding_method"] = "focus" 
            #store result
            data.append(res)

def splice_focus_embedding_similarity(model, ram_model, metrics, target_layer, hq_pth:str, rec_pth:str, data:list, weights, do_gradcam=True, rank=None):
    #set the weights
    model.splice_focus_img_embedder.weights = weights
    for metric in metrics:
        res = compute_embedding_sim_metrics(model.splice_focus_img_embedder ,target_layer ,hq_pth ,rec_pth, metric, do_gradcam=do_gradcam)
        #add approach name
        res["embedding_method"] = "splice_focus" 
        res["rank"] = rank

        #store result
        data.append(res)
    

