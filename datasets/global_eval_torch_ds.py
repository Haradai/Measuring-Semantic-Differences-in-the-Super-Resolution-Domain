from torch.utils.data import Dataset
import pandas as pd

class UserStudyScores(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sr_img = row["img_names"]
        hq_img = sr_img.split("_")[-1]

        sr_img = "150_clip+koniq_set/SR/" + sr_img
        hq_img = "150_clip+koniq_set/HQ/" + hq_img
        hq_img = hq_img.replace(".png",".jpg")

        img_a = model.processor(Image.open(sr_img).convert('RGB'))
        img_b = model.processor(Image.open(hq_img).convert('RGB'))
        cosine = float(row['userStudyScores'])
        return (img_a, img_b), cosine

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
import math
import torch.nn.functional as F

class ImpaintSDD_contrastive_dataset_clip(Dataset):
    def __init__(self, instances_csv:str):
        instances_df = pd.read_csv("ImpaintSDD_COCO2017Val_dreamSD_set1/instances.csv")

        #load negative images paths:
        negatives = np.array(os.listdir("ImpaintSDD_COCO2017Val_dreamSD_set1/impainted_proposals"))
        negative_indexes = np.array([int(neg.split("_")[0]) for neg in negatives])

        #load positive images paths:
        positives = np.array(os.listdir("ImpaintSDD_COCO2017Val_dreamSD_set1/positive_pairs"))
        positive_ids = np.array([int(pos.split("_")[0]) for pos in positives])

        self.IM_WIDTH = 224

        self.samples = []
        for j, row in  instances_df.iterrows():
            #print(positives[np.where(positive_ids == row["id"])[0]])
            negative_paths = negatives[np.where(negative_indexes == j)[0]]
            positive_paths = positives[np.where(positive_ids == row["id"])[0]]

            #skip instances that don't have posiotive or negative images becasue they have been discarded
            if len(negative_paths) == 0 or len(positive_paths) == 0:
                continue
            
            negative_paths = ["ImpaintSDD_COCO2017Val_dreamSD_set1/impainted_proposals/" + im for im in negative_paths]
            positive_paths = ["ImpaintSDD_COCO2017Val_dreamSD_set1/positive_pairs/" + im for im in positive_paths] 

            samp = {
                "id" : row["id"],
                "bbox" : self.parse_bboxstring(row["bbox"]),
                "negative_paths" : negative_paths,
                "positive_paths" : positive_paths,
                "gt_im_pth" : row["gt_image_path"]

            }
            self.samples.append(samp)

        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        positive_ims = [Image.open(pth) for pth in self.samples[idx]["positive_paths"]] 
        negative_ims = [Image.open(pth) for pth in self.samples[idx]["negative_paths"]]

        #load gt and resize (for some reason they have a different size and an error throws in the preprocessor)
        gt_im = Image.open(self.samples[idx]["gt_im_pth"])
        # Resize gt_image to match dimensions of other images
        gt_im = gt_im.resize(positive_ims[0].size, Image.BILINEAR)
        positive_ims.append(gt_im)
        
        if len(positive_ims) == 0:
            print("WTF!")
            print(idx)
            print(self.samples[idx])
        labels = [1]*len(positive_ims) + [0]*len(negative_ims)

        ims = self.processor(
            images= positive_ims + negative_ims,
            padding=True,
            do_center_crop=False,
            do_resize=True,
            size={"shortest_edge": 224},
            return_tensors="pt"
        )
        
        # Image sizes
        im_w = ims["pixel_values"].shape[3]
        im_h = ims["pixel_values"].shape[2]

        # Will perform a 255x255 crop around the image
        bbox_init = self.samples[idx]["bbox"]
        # Scale the bbox values for the rescaled image
        rescale_factor = im_h / positive_ims[0].size[1] 
        
        bbox_init = [math.ceil(val*rescale_factor) for val in bbox_init]
        
        # Extract initial coordinates
        x0 = bbox_init[0]
        y0 = bbox_init[1]
        width = bbox_init[2]
        height = bbox_init[3]
        
        # Calculate end coordinates
        x1 = x0 + width
        y1 = y0 + height
        
        # Calculate how much we need to grow to reach 255x255
        width_to_grow = self.IM_WIDTH - width
        height_to_grow = self.IM_WIDTH - height
        
        # Determine which side to grow based on proximity to edges
        # Handle width (x-dimension)
        left_distance = x0
        right_distance = im_w - x1
        
        if left_distance <= right_distance:
            # Right edge is farther, grow more on right
            x1_grow = min(width_to_grow, right_distance)
            x1 += x1_grow
            # Use remaining growth for left if needed
            x0_grow = width_to_grow - x1_grow
            x0 = max(0, x0 - x0_grow)
        else:
            # Left edge is farther, grow more on left
            x0_grow = min(width_to_grow, left_distance)
            x0 -= x0_grow
            # Use remaining growth for right if needed
            x1_grow = width_to_grow - x0_grow
            x1 = min(im_w, x1 + x1_grow)
        
        # Handle height (y-dimension)
        top_distance = y0
        bottom_distance = im_h - y1
        
        if top_distance <= bottom_distance:
            # Bottom edge is farther, grow more on bottom
            y1_grow = min(height_to_grow, bottom_distance)
            y1 += y1_grow
            # Use remaining growth for top if needed
            y0_grow = height_to_grow - y1_grow
            y0 = max(0, y0 - y0_grow)
        else:
            # Top edge is farther, grow more on top
            y0_grow = min(height_to_grow, top_distance)
            y0 -= y0_grow
            # Use remaining growth for bottom if needed
            y1_grow = height_to_grow - y0_grow
            y1 = min(im_h, y1 + y1_grow)
        
        #final check on the crops to ensure 255x255
        # Apply the crop to the processed tensors
        pixel_vals = ims["pixel_values"][:, :, math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)]

        # Ensure the final cropped patch is exactly 255x255 (sometimes is off by 1 pixel)
        pixel_vals = F.interpolate(pixel_vals, size=( self.IM_WIDTH,  self.IM_WIDTH), mode='bilinear', align_corners=False)

        return pixel_vals, torch.tensor(labels), torch.tensor([idx]*len(labels))

    
    def parse_bboxstring(self, bboxs:str) -> list[int]:
        bbox = bboxs.split(",")
        bbox[0] = bbox[0][1:]
        bbox[-1] = bbox[-1][:-1]
        bbox = [float(v) for v in bbox]
        bbox[0] = int(math.floor(bbox[0]))
        bbox[1] = int(math.floor(bbox[1]))
        bbox[2] = int(math.ceil(bbox[2]))
        bbox[3] = int(math.ceil(bbox[3]))
        return bbox