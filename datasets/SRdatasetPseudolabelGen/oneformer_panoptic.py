from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import os
from tqdm import tqdm
import pickle
import torch
import numpy as np
from scipy import ndimage
from PIL import Image

def load_model():
    # load processor for preprocessing the inputs
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = model.to("cuda")
    return processor, model


def reassign_disconnected_components(panoptic_map: torch.Tensor, ignore_background: bool = False) -> torch.Tensor:
    """
    Reassigns disconnected regions of the same instance ID to new unique IDs.

    Args:
        panoptic_map: 2D torch.Tensor with instance IDs.
        ignore_background: If True, skips background ID 0.

    Returns:
        2D torch.Tensor with reassigned IDs for disconnected regions.
    """
    np_map = panoptic_map.cpu().numpy()
    result = np.zeros_like(np_map)
    current_id = 1  # Start labeling from 1

    unique_ids = np.unique(np_map)
    if ignore_background:
        unique_ids = unique_ids[unique_ids != 0]

    for inst_id in unique_ids:
        mask = (np_map == inst_id)
        labeled, num_features = ndimage.label(mask)
        for lab in np.arange(1,num_features+1): #avoid labiling in ouput 0
            lab_msk = labeled == lab
            result[lab_msk] = lab + current_id
        current_id += num_features

    return torch.tensor(result, dtype=panoptic_map.dtype, device=panoptic_map.device)

def filter_small_components(panoptic_map: torch.Tensor, area_threshold_percent: float = 1.0) -> torch.Tensor:
    """
    Removes small regions by reassigning them to the most common neighboring label.
    Iterates until no more small components remain.

    Args:
        panoptic_map: 2D torch.Tensor with instance IDs.
        area_threshold_percent: Threshold (percentage of image) below which components are considered small.

    Returns:
        Updated 2D torch.Tensor with small regions eroded into neighboring regions.
    """
    np_map = panoptic_map.cpu().numpy()
    h, w = np_map.shape
    total_area = h * w
    min_area = int((area_threshold_percent / 100.0) * total_area)

    result = np_map.copy()
    
    # Keep iterating until no more changes are made
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        unique_ids = np.unique(result)  # Recalculate unique IDs each iteration
        
        for inst_id in unique_ids:
            component = (result == inst_id)
            if np.sum(component) < min_area:
                # Find neighboring labels
                dilated = ndimage.binary_dilation(component)
                neighbors = np.unique(result[dilated & ~component])
                neighbors = neighbors[neighbors != inst_id]
                
                if len(neighbors) > 0:
                    # Assign to the most frequent neighboring label
                    counts = [(n, np.sum(result[dilated & ~component] == n)) for n in neighbors]
                    only_counts = [c[1] for c in counts]
                    max_idx = np.argmax(only_counts)
                    new_lab = counts[max_idx][0]
                    result[component] = new_lab
                    changed = True  # Mark that we made a change
        
        # Safety check to avoid infinite loops
        if iteration > 100:  # Adjust this limit as needed
            print(f"Warning: Stopped after {iteration} iterations")
            break
    
    print(f"Converged after {iteration} iterations")
    return torch.tensor(result, dtype=panoptic_map.dtype, device=panoptic_map.device)

def compute_panoptic(pil_img, processor, model) -> torch.tensor:
    inputs = processor(pil_img, ["panoptic"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # you can pass them to processor for panoptic postprocessing
    predicted_panoptic_map = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(pil_img.height, pil_img.width)]
    )[0]["segmentation"]

    new_pana = reassign_disconnected_components(predicted_panoptic_map)
    new_pana = filter_small_components(new_pana,area_threshold_percent=0.5)
    new_pana = reassign_disconnected_components(new_pana) #to reassign indexes

    return new_pana

def generate_panoptics_pickls(img_folder:str, processor, model):
    """
    Generate masks for all images in a folder and save them as pickle files.
    
    Args:
        img_folder (str): Path to the folder containing images.
        mask_generator (SamAutomaticMaskGenerator): The SAM model for mask generation.
        min_area (float): Minimum area for masks to be considered.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(img_folder, "general_panoptic")
    os.makedirs(output_dir, exist_ok=True)


    # Iterate through all images in the folder
    for img_name in tqdm(os.listdir(img_folder), desc="Extracting masks"): #all images have same filename so we can iterate just the firts folder
        if img_name.endswith(('.png', '.jpg', '.jpeg')):

            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path)

            pana = compute_panoptic(img, processor, model)
            
            # Save masks to pickle file
            pickle_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(pana, f)


if __name__ == "__main__":
    processor, model = load_model()
    folders = [
        "koniq-10k_sr/HQ",
        "koniq-10k_sr/LQ_x4_bsrgan_out",
        "koniq-10k_sr/LQ_x4_swinir_out",
        "koniq-10k_sr/LQ_x4_seesr_out",
        "koniq-10k_sr/LQ_x4_pasd_out",
        "koniq-10k_sr/LQ_x4_stablesr_out",
        "koniq-10k_sr/LQ_x4_degfac_0.3_bsrgan_out",
        "koniq-10k_sr/LQ_x4_degfac_0.3_swinir_out",
        "koniq-10k_sr/LQ_x4_degfac_0.3_seesr_out",
        "koniq-10k_sr/LQ_x4_degfac_0.3_pasd_out",
        "koniq-10k_sr/LQ_x4_degfac_0.3_stablesr_out",
        "koniq-10k_sr/LQ_x4_degfac_0.7_bsrgan_out",
        "koniq-10k_sr/LQ_x4_degfac_0.7_swinir_out",
        "koniq-10k_sr/LQ_x4_degfac_0.7_seesr_out",
        "koniq-10k_sr/LQ_x4_degfac_0.7_pasd_out",
        "koniq-10k_sr/LQ_x4_degfac_0.7_stablesr_out",
    ]
    for f in folders:
        generate_panoptics_pickls(f, processor, model)
        print(f"Panoptic segmentation masks generated and saved for {f}")