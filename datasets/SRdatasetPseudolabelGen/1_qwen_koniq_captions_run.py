from modelscope import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import pickle
from PIL import Image


def load_qwen_model():
    model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("qwen/Qwen2-VL-7B-Instruct")

    return model, processor


def caption_batch(image_paths, processor, model):
    """Caption a batch of images"""
    # Prepare messages for batch processing
    batch_messages = []
    for img_path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        batch_messages.append(messages)
    
    # Process all messages in the batch
    texts = []
    all_image_inputs = []
    all_video_inputs = []
    
    for messages in batch_messages:
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        
        image_inputs, video_inputs = process_vision_info(messages)
        all_image_inputs.extend(image_inputs if image_inputs else [])
        all_video_inputs.extend(video_inputs if video_inputs else [])
    
    # Process the batch
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts


def caption_maskim(mask_im, processor, model):
    """Single image captioning (kept for backwards compatibility)"""
    return caption_batch([mask_im], processor, model)[0]


def caption_folder_batch(model, processor, pth, batch_size=4):
    """Caption all images in a folder using batch processing"""
    imgs = os.listdir(pth)
    imgs = [im for im in imgs if im.endswith((".png", ".jpg"))]
    
    # Create full paths
    img_paths = [os.path.join(pth, im) for im in imgs]
    
    # Process in batches
    all_captions = []
    with open(pth + "/wholeim_captions_batch.txt", "w") as f:
        for i in tqdm(range(0, len(img_paths), batch_size), desc=f"Processing {pth}"):
            batch_paths = img_paths[i:i + batch_size]
            batch_names = imgs[i:i + batch_size]
            
            captions = caption_batch(batch_paths, processor, model)
            
            for name, caption in zip(batch_names, captions):
                #print(f"{name}: {caption}")
                f.write(f"Image: {name}\n")
                f.write(f"Caption: {caption}\n\n\nNEXT_CAPTION\n\n\n")
                all_captions.append((name, caption))
            '''
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Fall back to individual processing for this batch
            for path, name in zip(batch_paths, batch_names):
                try:
                    caption = caption_maskim(path, processor, model)
                    print(f"{name}: {caption}")
                    f.write(f"Image: {name}\n")
                    f.write(f"Caption: {caption}\n\n\nNEXT_CAPTION\n\n\n")
                    all_captions.append((name, caption))
                except Exception as e2:
                    print(f"Error processing {name}: {e2}")
                    f.write(f"Image: {name}\n")
                    f.write(f"Caption: ERROR - {str(e2)}\n\n\nNEXT_CAPTION\n\n\n")
            '''
    return all_captions


def caption_folder(model, processor, pth):
    """Original single-image processing function (kept for backwards compatibility)"""
    imgs = os.listdir(pth)
    imgs = [im for im in imgs if im.endswith((".png", ".jpg"))]
    with open(pth + "/wholeim_captions.txt", "a") as f:
        for im in tqdm(imgs):
            cap = caption_maskim(pth + "/" + im, processor, model)
            print(cap)
            f.write(cap + "\n\n\nNEXT_CAPTION\n\n\n")


if __name__ == "__main__":
    # Load the model and processor
    print("Loading model...")
    model, processor = load_qwen_model()
    print("Model loaded successfully!")

    folders = [
        "koniq-10k_sr/HQ",
        "koniq-10k_sr/LQ_x4_bsrgan_out",
        "koniq-10k_sr/LQ_x4_swinir_out",
        "koniq-10k_sr/LQ_x4_seesr_out",
        "koniq-10k_sr/LQ_x4_pasd_out",
        "koniq-10k_sr/LQ_x4_stablesr_out"
    ]

    # Batch processing (recommended)
    batch_size = 10 # Adjust based on your GPU memory
    for pth in folders:
        print(f"\nProcessing folder: {pth}")
        if os.path.exists(pth):
            captions = caption_folder_batch(model, processor, pth, batch_size=batch_size)
            print(f"Completed {pth}: {len(captions)} images processed")
        else:
            print(f"Warning: Folder {pth} does not exist")

    # Alternative: Single image processing (original method)
    # Uncomment the lines below if you prefer the original single-image method
    """
    for pth in folders:
        if os.path.exists(pth):
            imgs = os.listdir(pth)
            imgs = [im for im in imgs if im.endswith((".png",".jpg"))]
            with open(pth + "/wholeim_det_caps.txt", "a") as f:
                for im in tqdm(imgs):
                    cap = caption_maskim(pth + "/" + im, processor, model)
                    print(cap)
                    f.write(cap + "\n\n\nNEXT_CAPTION\n\n\n")
    """