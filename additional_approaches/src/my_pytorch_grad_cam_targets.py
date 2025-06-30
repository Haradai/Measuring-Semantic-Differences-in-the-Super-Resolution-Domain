'''
We add our own functions here instead to in pytorch_grad_cam/utils/model_targets.py
'''
import torch 
import torch.nn.functional as F

class MSEOutputTarget:
    def __init__(self, target):
        self.target = target
        pass

    def __call__(self, model_output,reduction="mean"):
        return torch.nn.functional.mse_loss(model_output, self.target,reduction=reduction)

class CosineSimilarityOutputTarget:
    def __init__(self, target):
        self.target = target
        # Normalize target once during initialization for efficiency
        self.normalized_target = F.normalize(target, dim=-1)
    
    def __call__(self, model_output, reduction="mean"):

        # Normalize model output
        normalized_output = F.normalize(model_output, dim=-1)
        
        # Calculate cosine similarity
        cos_sims = self.normalized_target * normalized_output
        if cos_sims.ndim == 2:
            cos_sims = cos_sims.sum(dim=1)
        else:
            cos_sims = cos_sims.sum(dim=0)
        
        # Apply reduction
        if reduction == "mean":
            return torch.mean(cos_sims)
        elif reduction == "sum":
            return torch.sum(cos_sims)
        elif reduction == "none":
            return cos_sims
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")
        
class VectorSumOutputTarget:
    def __init__(self, target):
        pass

    def __call__(self, model_output,reduction="mean"):
        return model_output.sum()