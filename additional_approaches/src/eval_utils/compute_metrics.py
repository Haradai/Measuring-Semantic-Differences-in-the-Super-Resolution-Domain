import torch 
from src.gcam_utils import compute_grad_cam
from src.my_pytorch_grad_cam_targets import VectorSumOutputTarget
import pyiqa
import torch.nn.functional as F

def compute_similarity_matrix(vectors, loss_class, dim_mean=False):
    '''
    Given a list of embeddings and a loss class compute similarity matrix
    '''
    n = len(vectors)
    similarity_matrix = torch.zeros((n, n))
    
    for i in range(n):
        metric_obj = loss_class(vectors[i].repeat(n,1))
        out =  metric_obj(vectors,reduction="none").detach().cpu()
        if dim_mean:
            out = out.mean(dim=1)

        similarity_matrix[i,:] = out
    
    return similarity_matrix


psnr_metric = pyiqa.create_metric('psnr',device="cuda")
ssim_metric = pyiqa.create_metric('ssim',device="cuda")
lpips_metric = pyiqa.create_metric('lpips', device="cuda")


###More metrics to compute on the grad cams

def binarize_top_percent(tensor: torch.Tensor, top_percent: float = 0.9) -> torch.Tensor:
    """
    Binarizes a heatmap tensor by setting the top `top_percent` values to 1 and the rest to 0.
    
    Args:
        tensor (torch.Tensor): Input tensor (heatmap).
        top_percent (float): The proportion of values to binarize (e.g., 0.8 for top 0.8%).

    Returns:
        torch.Tensor: Binarized tensor with 1s for top `top_percent` values and 0s elsewhere.
    """
    # Flatten the tensor to find the top values
    threshold = torch.quantile(tensor.float(), 1 - top_percent)
    return (tensor >= threshold).float()

def gcam_compute_overlap_metrics(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """
    Computes overlap metrics between two binarized tensors A and B.

    Args:
        A (torch.Tensor): First tensor (heatmap).
        B (torch.Tensor): Second tensor (heatmap).

    Returns:
        tuple: Three metrics:
            - % of A overlapping in B.
            - % of B overlapping in A.
            - % of overlap relative to total number of pixels.
    """
    # Binarize both tensors
    B = F.interpolate(B.unsqueeze(0).unsqueeze(0), size=A.shape, mode='bilinear', align_corners=False)
    B = B.squeeze(0).squeeze(0)
    bin_A = binarize_top_percent(A)
    bin_B = binarize_top_percent(B)
    
    # Compute overlaps
    overlap = (bin_A * bin_B).sum().item()
    total_A = bin_A.sum().item()
    total_B = bin_B.sum().item()
    total_pixels = bin_A.numel()

    # Calculate metrics
    percent_A_in_B = (overlap / total_A) * 100 if total_A > 0 else 0
    percent_B_in_A = (overlap / total_B) * 100 if total_B > 0 else 0
    percent_total_overlap = (overlap / total_pixels) * 100

    return percent_A_in_B, percent_B_in_A, percent_total_overlap


def compute_embedding_sim_metrics(model, layer, imgHQ_pth:str, imgRec_pth:str, metric, do_gradcam=True):
    ''' 
    Embedding of img A
    Embedding of img B
    Then take distance for each metric
    Compute gradcams
    Return results
    
    Returns:
    Image similarity metrics -> (MSE, Cosine sim)
    Grad cams of each similarity metric
    '''
    def evaluate_grad_cam(gradcam:torch.tensor, refmap:torch.tensor, function) -> dict:
        '''
        Will compute the scores of the grad cam to a reference map.
        '''
        # Resize saliency map to model image size (gradcam size)
        # Assuming your tensor is named 'tensor'
        resized_refmap = F.interpolate(refmap.unsqueeze(0).unsqueeze(0), size=gradcam.shape, mode='bilinear', align_corners=False)
        
        return function(gradcam.unsqueeze(0).unsqueeze(0).to("cuda"), resized_refmap.to("cuda"))[0].item()


    hq_img = model.load_img(imgHQ_pth)
    rec_img = model.load_img(imgRec_pth)

    #reference img embedding
    hq_emb = model(hq_img)
    rec_emb = model(rec_img) 
    #we'll store everything here, also add hq and rec pth to data
    data = {"hq_pth" : imgHQ_pth,
            "rec_pth" : imgRec_pth,
            "metric" : metric.__name__}

    metric_obj = metric(hq_emb)
    sim =  metric_obj(rec_emb,reduction="mean").detach().cpu().item()
    #save metric
    data["score"] = sim

    #compute ssim, psnr and lpips
    data["ssim"] = ssim_metric(imgHQ_pth,imgRec_pth).item()
    data["psnr"] = psnr_metric(imgHQ_pth,imgRec_pth).item()
    data["lpips"] = lpips_metric(imgHQ_pth,imgRec_pth).item()
    
    
    if do_gradcam:
        #compute grad cam
        gradcam = torch.tensor(compute_grad_cam(model,metric,layer,rec_img,hq_emb))
        #load saliency map
        path_parts = imgHQ_pth.split("/")
        path_parts.insert(1, "SaliencyMaps")
        saliency_path = "/".join(path_parts)
        saliency_path = saliency_path.split(".")[0] + ".pt"
        saliency_map = torch.load(saliency_path)

        #compute scores of the gradcam
        data["gcam_psnr_saliency"] = evaluate_grad_cam(gradcam, saliency_map, psnr_metric)
        data["gcam_ssim_saliency"] = evaluate_grad_cam(gradcam, saliency_map, ssim_metric)

        #compute gradcam of the hq img embedding and evaluate diff gradcam on that too
        hq_emb_gcam = torch.tensor(compute_grad_cam(model,VectorSumOutputTarget,layer,hq_img,None))

        data["gcam_psnr_hqgcam"] = evaluate_grad_cam(gradcam, hq_emb_gcam, psnr_metric)
        data["gcam_ssim_hqgcam"] = evaluate_grad_cam(gradcam, hq_emb_gcam, ssim_metric)

        #Also add the score of embedding gradcam and saliency
        data["hqgcam_psnr_saliency"] = evaluate_grad_cam(hq_emb_gcam,saliency_map,psnr_metric)
        data["hqgcam_ssim_saliency"] = evaluate_grad_cam(hq_emb_gcam,saliency_map,ssim_metric)

        #compute overlap metrics
        percent_A_in_B, percent_B_in_A, percent_total_overlap = gcam_compute_overlap_metrics(hq_emb_gcam,saliency_map)
        data["hqgcam_vsoverlap_saliency"] = percent_A_in_B
        data["saliency_vsoverlap_hqgcam"] = percent_B_in_A
        data["hqgcam_overlap_saliency"] = percent_total_overlap

        #compute overlap metrics
        percent_A_in_B, percent_B_in_A, percent_total_overlap = gcam_compute_overlap_metrics(gradcam,saliency_map)
        data["gradcam_vsoverlap_saliency"] = percent_A_in_B
        data["saliency_vsoverlap_gradcam"] = percent_B_in_A
        data["gradcam_overlap_saliency"] = percent_total_overlap

        #compute overlap metrics
        percent_A_in_B, percent_B_in_A, percent_total_overlap = gcam_compute_overlap_metrics(hq_emb_gcam,gradcam)
        data["hqgcam_vsoverlap_gradcam"] = percent_A_in_B
        data["gradcam_vsoverlap_hqgcam"] = percent_B_in_A
        data["hqgcam_overlap_gradcam"] = percent_total_overlap
    
    return data