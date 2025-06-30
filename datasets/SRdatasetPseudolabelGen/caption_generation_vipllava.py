from ViPLLaVA.llava.model.builder import load_pretrained_model
from ViPLLaVA.llava.mm_utils import get_model_name_from_path
from ViPLLaVA.llava.eval.run_llava import eval_model_modded
import os
from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image
import cv2
import torch

def load_vip_llava_model():
    model_path = "captioning_dataset_src/ViPLLaVA/vip-llava-7b-refcocog-ft"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    return model_path, tokenizer, model, image_processor, context_len

def draw_mask_contour(image_path, mask):
    """
    Draw the contour of a mask on the image.
    
    Args:
        image_path (str): Path to the original image
        mask (np.ndarray): Binary mask for the object
    
    Returns:
        PIL.Image: Image with contour drawn
    """
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)

    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours in magenta
    contour_color = (255, 0, 255)  # BGR
    cv2.drawContours(np_img, contours, -1, contour_color, thickness=3)

    return Image.fromarray(np_img), "magenta"

def caption_with_mask_contour(image_path, contour_color, model_path, tokenizer, model, image_processor, context_len):
    """
    Generate a caption for the image with contour overlay.
    """
    prompt = f"Please describe the object outlined in {contour_color}."
    
    args = type('Args', (), {
        "query": prompt,
        "model_path": model_path,
        "image_file": image_path,
        "conv_mode": None,
        "model_base": None,
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "sep": ",",
    })()
    
    caption = eval_model_modded(args, tokenizer, model, image_processor, context_len)
    return caption

def panoptic_to_masks(panoptic_tensor: torch.Tensor):
    """
    Converts a panoptic map tensor with integer values into a list of binary masks (numpy arrays),
    one for each unique integer in the panoptic tensor.
    
    Args:
        panoptic_tensor (torch.Tensor): 2D tensor with integer values representing panoptic segments.
        
    Returns:
        List[np.ndarray]: List of binary masks as numpy arrays (dtype=bool), one per unique segment id.
    """
    unique_ids = torch.unique(panoptic_tensor)
    masks = []
    
    for uid in unique_ids:
        mask = (panoptic_tensor == uid).cpu().numpy()  # boolean mask as numpy array
        masks.append(mask)
        
    return masks


def process_images(imgs_folder: str, model_path, tokenizer, model, image_processor, context_len ):
    """
    Process images to generate captions for mask contours.
    """
    
    ims = [im for im in os.listdir(imgs_folder) if im.endswith(('.png', '.jpg','.jpeg'))]
    img_names = [im.split('.')[0] for im in ims]
    suffix = ims[0].split(".")[-1]
    print(len(ims))
    for im in img_names:
        im_pth = os.path.join(imgs_folder, im + '.' + suffix)

        gen_pkl_pth = os.path.join(imgs_folder, "general_panoptic", im + '.pkl')
        
        with open(gen_pkl_pth, 'rb') as f:
            general = pickle.load(f)
        

        #####process general
        captions = []
        masks = panoptic_to_masks(general)
        for mask in tqdm(masks, desc=f"Generating captions for {im}"):
            image_with_contour, contour_color = draw_mask_contour(im_pth, mask)
            temp_img_path = "temp_img_with_contour.png"
            image_with_contour.save(temp_img_path)

            caption = caption_with_mask_contour(temp_img_path, contour_color, model_path, tokenizer, model, image_processor, context_len)
            captions.append(caption)
            print(caption)

        os.makedirs(os.path.join(imgs_folder, "general_panoptic"), exist_ok=True)
        caption_path = os.path.join(imgs_folder, "general_panoptic", f"{im}.txt")
        
        with open(caption_path, "w") as f:
            for caption in captions:
                f.write(caption + "\n")
        
        print(f"Captions for {im} saved to {caption_path}")

        if os.path.exists("temp_img_with_contour.png"):
            os.remove("temp_img_with_contour.png")

if __name__ == "__main__":
    # Load the model and processor
    print("Loading ViP-LLaVA model...")
    model_path, tokenizer, model, image_processor, context_len = load_vip_llava_model()
    
    folders = [
        "koniq-10k_sr/HQ",
        "koniq-10k_sr/LQ_x4_bsrgan_out",
        "koniq-10k_sr/LQ_x4_swinir_out",
        "koniq-10k_sr/LQ_x4_seesr_out",
        "koniq-10k_sr/LQ_x4_pasd_out",
        "koniq-10k_sr/LQ_x4_stablesr_out"
    ]
    
    for f in folders:
       # Process the images to generate captions
        process_images(f, model_path, tokenizer, model, image_processor, context_len)
        print(f"Panoptic segmentation masks generated and saved for {f}")