
from torch.utils.data import Dataset
import pandas as pd
# Create an instance of your model with parameters coming from the sweep
from src.models.clip_based.lpips_like import CLIP_lpips_stages_cnn
import torch
from PIL import Image

# Load this only to have the image processor available
model = CLIP_lpips_stages_cnn(
    clip_name="resnet50_clip.openai",  # or another model identifier from timm if applicable
    depth=1,
    device="cpu"
)

class UserStudyScores(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sr_img = row["Super Resolution Image"]
        hq_img = sr_img.split("_")[-1]

        sr_img = "150_clip+koniq_set/SR/" + sr_img
        hq_img = "150_clip+koniq_set/HQ/" + hq_img
        hq_img = hq_img.replace(".png",".jpg")

        img_a = model.processor(Image.open(sr_img).convert('RGB'))
        img_b = model.processor(Image.open(hq_img).convert('RGB'))
        if row["Answer"] == "Yes":
            cosine = 1.0
        else:
            cosine = 0.0

        return (img_a, img_b), cosine


import wandb
import torch
import itertools
from tqdm import tqdm
from transformers import CLIPModel

def train_contrastive(model, train_dataloader, val_dataloader, device, epochs=10, lr=1e-4, model_name="clip_resnet50_stages_cnn"):
    
    #clip = CLIPLoRAModule(model_name="openai/clip-vit-base-patch32", lora_rank=8).to(device)
    # Load the base CLIP model
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, ((im0, im1), target_cosine) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            outs = model(im0.to(device),im1.to(device))
            loss = criterion(outs, target_cosine.to(device))

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
            for (im0, im1), target_cosine in val_dataloader:
                outs = model(im0.to(device),im1.to(device))
                loss = criterion(outs, target_cosine.to(device))
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Eval Loss: {avg_eval_loss:.4f}")

        #save model checkpoint
        model.save_model(model_name + ".pt")

        # Log the validation loss for the epoch
        wandb.log({"eval_loss_epoch": avg_eval_loss, "epoch": epoch + 1})

    wandb.finish()



from src.models.clip_based.lpips_like import CLIP_lpips_stages_cnn
# ========================================================
# STEP 1: Define the sweep configuration
sweep_configuration = {
    "method": "grid",  # You can change this method as needed (e.g., "bayes" for Bayesian optimization)
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "depth": {"values": [1,2,3]},
        "lr": {"values": [1e-4]}
    },
}
# ========================================================

# STEP 2: Define the sweep training function
def sweep_train():

    wandb.init(project="userstudy_regressor_yesnos")

    # Now wandb.config is available
    config = wandb.config
    run_name = f"CLIP_RESNET_STAGES_DEPTH:{config.depth}_yesno"

    # You can optionally rename the run after init
    wandb.run.name = run_name
    
    # Define your device, dataloaders, etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create an instance of your model with parameters coming from the sweep
    model = CLIP_lpips_stages_cnn(
    clip_name="resnet50_clip.openai",
    depth = config.depth,  # Use the min_caps from the sweep config
    device="cuda"
    )

    # Create datasets
    whole_dataset = UserStudyScores("150_clip+koniq_set/filtered_answers.csv")

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
        imgs, cosines = zip(*batch)
        img_a, img_b = zip(*imgs)
        img_a = torch.stack(img_a)
        img_b = torch.stack(img_b)
        cosines = torch.tensor(cosines)
        return (img_a, img_b), cosines
            
    from torch.utils.data import DataLoader
    # Create data loaders
    batch_size = 5  # Adjust based on your memory constraints
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
        epochs=30,
        lr=config.lr,
        model_name = run_name 
    )

# ========================================================
# STEP 3: Start the sweep
import wandb

sweep_id = wandb.sweep(sweep=sweep_configuration, project="userstudy_regressor_yesno")
wandb.agent(sweep_id, function=sweep_train)
