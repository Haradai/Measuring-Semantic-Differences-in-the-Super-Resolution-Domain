
from torch.utils.data import Dataset
import pandas as pd
import pickle
from PIL import Image
import torch
import torchvision

import timm
import torch.nn as nn
from icecream import ic

import wandb
import torch
from tqdm import tqdm


from models.datasets import KoNiqPairsDataset_maps
from models.models import CLIP_lpips_Unet

import multiprocessing
import os
from torch.utils.data import DataLoader

def train_contrastive(model, train_dataloader, val_dataloader, device, epochs=10, lr=1e-4, model_name="clip_lpips_unet"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, ((im0, im1), cosmap) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            outs = model(im0.to(device),im1.to(device))
            loss = criterion(outs, cosmap.to(device).float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log the batch loss to wandb
            wandb.log({"train_loss_batch": loss.item(), "epoch": epoch + 1, "batch_idx": batch_idx})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Log the average train loss for the epoch
        wandb.log({"train_loss_epoch": avg_train_loss, "epoch": epoch + 1})

        # Evaluation pass (no gradients, model in eval mode)
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for (im0, im1), cosmap in val_dataloader:
                outs = model(im0.to(device),im1.to(device))
                loss = criterion(outs, cosmap.to(device).float())
                eval_loss += loss.item()

            #save model to path
            model.save_model(model_name + ".pt")

        avg_eval_loss = eval_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Eval Loss: {avg_eval_loss:.4f}")

        # Log the validation loss for the epoch
        wandb.log({"eval_loss_epoch": avg_eval_loss, "epoch": epoch + 1})

    wandb.finish()



# ========================================================
# STEP 1: Define the sweep configuration
sweep_configuration = {
    "method": "grid",  # You can change this method as needed (e.g., "bayes" for Bayesian optimization)
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "min_caps": {"values": [2,4,8,16]},
        "only_hq": {"values": [True,False]},
        "lr": {"values": [1e-4]},
        "lora_rank": {"values": [None, 32, "full"]},
        "threshold": {"values": [None, 0.4,0.9]},
        "pretrained_bckbn" : {"values" : ["clip, imgnet"]}
    },
}
# ========================================================

# STEP 2: Define the sweep training function
def sweep_train():
    wandb.init(project="semantic_maps_pseudo_bin")

    # Now wandb.config is available
    config = wandb.config
    run_name = f"CLIP_LPIPS_UNET_HQ:{config.only_hq}_MINCAPS:{config.min_caps}_lora:{config.lora_rank}_threshold:{config.threshold}"

    # You can optionally rename the run after init
    wandb.run.name = run_name
    
    # Define your device, dataloaders, etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create an instance of your model with parameters coming from the sweep
    model = CLIP_lpips_Unet(
    clip_name="resnet50_clip.openai",
    device="cuda",
    lora_rank=config.lora_rank
    )

    #Define dataset with config
    whole_dataset = KoNiqPairsDataset_maps(
        model_processor=model.processor,
        csv_path="koniq-10k_sr/cosine_maps/filt_refs.csv",
        only_hq=config.only_hq,
        imgamincaps=config.min_caps,
        threshold=config.threshold
        )
    
    from torch.utils.data import random_split
    # Split the dataset into training and testing sets
    train_ratio = 0.8  # 80% for training, 20% for testing

    # Calculate the size of each split
    dataset_size = len(whole_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Perform the split
    train_dataset, test_dataset = random_split(
        whole_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Example collate_fn (if not defined elsewhere)
    def collate_fn(batch):
        images, cosmap = zip(*batch)
        img_a, img_b = zip(*images)
        img_a = torch.stack(img_a)
        img_b = torch.stack(img_b)
        cosmap = torch.stack(cosmap)
        return (img_a, img_b), cosmap
            
    # Create data loaders
    batch_size = 80  # Adjust based on your memory constraints
    num_workers = 8  # Adjust based on your CPU cores

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Helps speed up data transfer to GPU
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Run training with hyperparameters from the config
    train_contrastive(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=60,
        lr=config.lr,
        model_name = run_name 
    )

# ========================================================
# STEP 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="semantic_maps_pseudo_bin")

def run_agent_on_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    wandb.agent(sweep_id, function=sweep_train)

if __name__ == "__main__":
    gpu_ids = [5,3]  # Specify the GPU IDs you want to use
    processes = []
    for gpu_id in gpu_ids:
        p = multiprocessing.Process(target=run_agent_on_gpu, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
