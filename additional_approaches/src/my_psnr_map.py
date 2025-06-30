import torch
import piq
import torch.nn.functional as F

def compute_psnr_map(X: torch.tensor, Y: torch.tensor, window_size: int = 11, device = torch.device("cpu")) -> torch.tensor:
    '''
    Compute SSIM map efficiently using batch processing and unfold.
    Assumes images come unbatched (single images, no batch dimension).
    '''
    assert X.shape == Y.shape, "X and Y should be the same shape!"
    
    # Parameters
    half_w = window_size // 2
    channels, height, width = X.shape
    
    # Add padding to X and Y
    X_padded = F.pad(X.unsqueeze(0), (half_w, half_w, half_w, half_w), mode='replicate').squeeze(0)
    Y_padded = F.pad(Y.unsqueeze(0), (half_w, half_w, half_w, half_w), mode='replicate').squeeze(0)
    
    # Unfold to extract patches
    X_patches = X_padded.unfold(1, window_size, 1).unfold(2, window_size, 1)
    Y_patches = Y_padded.unfold(1, window_size, 1).unfold(2, window_size, 1)

    # Reshape patches to prepare for SSIM calculation
    X_patches = X_patches.contiguous().view(channels, -1, window_size, window_size).permute(1, 0, 2, 3)
    Y_patches = Y_patches.contiguous().view(channels, -1, window_size, window_size).permute(1, 0, 2, 3)
    
    # Compute SSIM for each patch (batched)
    psnr_values = piq.psnr(X_patches.to(device), Y_patches.to(device), data_range=1.0, reduction="none")
    
    # Reshape SSIM values back to the original image size
    psnr_map = psnr_values.view(height, width)
    
    return psnr_map