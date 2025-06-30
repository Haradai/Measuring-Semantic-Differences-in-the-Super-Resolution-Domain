import torch
import torchvision.io
import pyiqa
from Koniqpp.model_wrapper import ImageQualityAssessor
import os

class ImageQualityMetrics:
    """A callable class for computing image quality metrics."""
    
    def __init__(self, koniq_model_path="Koniqpp/pretrained_model"):
        """
        Initialize the ImageQualityMetrics class.
        
        Args:
            koniq_model_path (str): Path to the Koniq++ pretrained model
        """
        self.lpips_metr = pyiqa.create_metric('lpips', device="cuda", as_loss=False)
        self.ssim_metr = pyiqa.create_metric('ssimc', device="cuda", as_loss=False)
        self.koniq = ImageQualityAssessor(koniq_model_path)
    
    def __call__(self, gt_pth, out_pth):
        """
        Compute image quality metrics between ground truth and output images.
        
        Args:
            gt_pth (str): Path to ground truth image
            out_pth (str): Path to output image
            
        Returns:
            dict: Dictionary containing SSIM, LPIPS, and Koniq++ IQA scores
        """
        # Load images and convert to float tensors
        gt_img = torchvision.io.read_image(gt_pth).float() / 255.0
        out_img = torchvision.io.read_image(out_pth).float() / 255.0
        
        # Add batch dimension (required by PIQ)
        gt_img = gt_img.unsqueeze(0)
        out_img = out_img.unsqueeze(0)
        
        # Compute SSIM (higher is better)
        ssim_score = self.ssim_metr(gt_img, out_img)[0]
        
        # Compute LPIPS (lower is better)
        lpips_score = self.lpips_metr(gt_img, out_img)[0]
        
        # Compute Koniq++ IQA score
        out_val = self.koniq(out_pth)
        
        return {
            'SSIM': ssim_score.item(),
            'LPIPS': lpips_score.item(),
            'Koniq++IQA': out_val["quality_score"],
            'Koniq++blur': out_val["blur"],
            'Koniq++artifacts': out_val["artifacts"],
            'Koniq++contrast': out_val["contrast"],
            'Koniq++color': out_val["color"]
        }

import pandas as pd
from pathlib import Path
from tqdm import tqdm
def process_image_folders(metrics_calc ,gt_folder, eval_folder, output_csv):
    """
    Process all images in folders and compute quality metrics.
    
    Args:
        gt_folder (str): Path to ground truth images folder
        eval_folder (str): Path to evaluation images folder  
        output_csv (str): Path where to save the CSV results
        koniq_model_path (str): Path to the Koniq++ pretrained model
        
    Returns:
        pd.DataFrame: DataFrame containing all computed metrics
    """
    
    # Get all image files from both folders
    gt_path = Path(gt_folder)
    eval_path = Path(eval_folder)
    
    # Common image extensions
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files from gt folder
    gt_files = {f.stem: f for f in gt_path.iterdir() 
                if f.is_file() and f.suffix.lower() in img_extensions}
    
    # Get all image files from eval folder
    eval_files = {f.stem: f for f in eval_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in img_extensions}
    
    # Find matching pairs
    common_names = set(gt_files.keys()) & set(eval_files.keys())
    
    if not common_names:
        raise ValueError("No matching image pairs found between the two folders")
    
    print(f"Found {len(common_names)} matching image pairs")
    
    # Process each pair
    results = []
    for i, name in tqdm(enumerate(sorted(common_names), 1),total=len(common_names),desc="computing metrics"):
        try:
            gt_file = gt_files[name]
            eval_file = eval_files[name]
        
            
            # Compute metrics
            metrics = metrics_calc(str(gt_file), str(eval_file))
            
            # Add metadata
            result = {
                'image_name': name,
                'gt_path': str(gt_file),
                'eval_path': str(eval_file),
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            # Add failed entry with NaN values
            result = {
                'image_name': name,
                'gt_path': str(gt_files[name]),
                'eval_path': str(eval_files[name]),
                'SSIM': float('nan'),
                'LPIPS': float('nan'),
                'Koniq++IQA': float('nan'),
                'Koniq++blur':  float('nan'),
                'Koniq++artifacts': float('nan'),
                'Koniq++contrast': float('nan'),
                'Koniq++color': float('nan'),
            }
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total images processed: {len(df)}")
    print(f"Successful: {df['SSIM'].notna().sum()}")
    print(f"Failed: {df['SSIM'].isna().sum()}")
    
    if df['SSIM'].notna().sum() > 0:
        print(f"\nMetric Averages:")
        print(f"SSIM: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
        print(f"LPIPS: {df['LPIPS'].mean():.4f} ± {df['LPIPS'].std():.4f}")
        print(f"Koniq++IQA: {df['Koniq++IQA'].mean():.4f} ± {df['Koniq++IQA'].std():.4f}")
    
    return df



import concurrent.futures

####RUN STAGE 1 ######

# Initialize metrics calculator (must be created inside each process due to CUDA context)
def process_folder(eval_folder):
    metrics_calc = ImageQualityMetrics("SR_dataset_src/Koniqpp/pretrained_model")
    hq_folder = "koniq-10k_sr/HQ"
    print(f"Processing folder: {eval_folder}")
    df = process_image_folders(
        metrics_calc,
        gt_folder=hq_folder,
        eval_folder=eval_folder,
        output_csv=os.path.join(eval_folder, "metrics.csv")
    )
    return eval_folder

folders = os.listdir("koniq-10k_sr")
folders = [f for f in folders if "out" in f] # filter to have only output folders
folders = [os.path.join("koniq-10k_sr", f) for f in folders] # add path to folders

# Use ProcessPoolExecutor for parallel processing (max 3 processes at a time)
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    list(executor.map(process_folder, folders))

###########################

