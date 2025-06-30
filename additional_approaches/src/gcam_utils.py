from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np

def compute_grad_cam(model, metric, target_layer, input_img, target_emb) -> torch.tensor:
    '''
    Returns the grad cam tensor at the model target layer, using the passed pytorch grad cam metric 
    '''
    def reshape_transform(tensor, height=model.patch_grid_size, width=model.patch_grid_size):
        result = tensor[:, 1 :  , :].reshape(tensor.size(0),height, width, tensor.size(2)) #put in grid like again

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = torch.movedim(result,3,1)
        return result
    
    with GradCAM(model=model, target_layers=[target_layer], reshape_transform= reshape_transform) as cam:
         targets = [metric(target_emb)]
         gradcam = cam(input_tensor=input_img, targets=targets)[0, :]
         return gradcam

def compute_gradcams_matrix(model, metric, target_layer, paths, embeddings) -> np.array:
    '''
    Given a list of torch tensor embeddings and a list of image paths to evaluate,
    it computes the combinatorial grad cam of the passed metric in a np array of shape (n_embeddings,n_paths,224, 224)
    '''
    n = len(embeddings)
    storer = np.empty((n,n,model.image_size,model.image_size))
    for i in range(n):
        img = model.load_img(paths[i])
        for j in range(n):
            gcam = compute_grad_cam(model, metric, target_layer, img, embeddings[j,:])
            storer[i,j] = gcam
    
    return storer

def create_overlayed_img_matrix(model, all_cams, paths):
    '''
    Returns a list of lists of PIL images with the overlayed cams. This is useful for visualization.
    '''
    all_overlayed_cams = []
    for i in range(all_cams.shape[0]):
        row = []
        for pth, j in zip(paths,range(all_cams.shape[1])):
            gcam = all_cams[i,j]
            img = model.load_img(pth)
            npy_norm_img = img.detach().cpu().squeeze(0)
            npy_norm_img = torch.movedim(npy_norm_img,0,2) #move channel dimension 
            npy_norm_img = npy_norm_img.numpy()
            npy_norm_img = np.clip(npy_norm_img, 0, 1)  # Clip values between 0 and 1
            cammed_img = show_cam_on_image(npy_norm_img, gcam, use_rgb=True) #here we are using a pytorch_grad_cam that we modified to return a PIL image
            row.append(cammed_img)#save PIL img
        
        all_overlayed_cams.append(row) #save row
    
    return all_overlayed_cams