import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import torch.nn.functional as F
import os
from itertools import combinations

class KoNiqPairsDataset_maps(Dataset):
    def __init__(self, model_processor, csv_path, only_hq = False, imgamincaps = 2, threshold = None):
        self.df = pd.read_csv(csv_path)
        self.threshold = threshold
        #filter by number of objects in the a image 
        self.df = self.df[self.df['ima_ncaps'] >= imgamincaps]

        if only_hq:
            # filter to only include high-quality images
            self.df = self.df[self.df['img_a_pth'].str.contains("HQ")]

        self.threshold = threshold

        self.model_processor = model_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_a = self.model_processor(Image.open(row['img_a_pth']))
        img_b = self.model_processor(Image.open(row['img_b_pth']))
        with open(row['out_paths'], 'rb') as f:
            cosmap = torch.tensor(pickle.load(f))
        
        if self.threshold is not None:
            #binarize the cosine map on threshold
            cosmap = (cosmap > self.threshold).float()
        
        #resize cosmap to match the input size of the model
        cosmap = F.interpolate(cosmap.unsqueeze(0).unsqueeze(0).float(), size=(img_a.shape[1], img_a.shape[2]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        return (img_a, img_b), cosmap

class ImpaintDS_maps(Dataset):
    def __init__(self, csv_path, model_processor):
        self.df = pd.read_csv(csv_path)
        folder_path = csv_path.split("/")[0]
        
        '''
        #remove discarted from the dataframe
        discarded = os.listdir(folder_path + "/discarded_proposals")

        to_disc = list(self.df["impainted_pth"])
        to_disc = [f.split("/")[-1] for f in to_disc]

        to_disc = set(to_disc) & set(discarded)
        self.df = self.df[~self.df["impainted_pth"].apply(lambda x: x.split('/')[-1] in to_disc)]

        '''
        #load also same image examples 
        pos_ims = os.listdir(folder_path + "/positive_pairs")

        pos_tripl = {}
        for im in pos_ims:
            key = im.split(".")[0].split("_")[0]
            if key not in pos_tripl.keys():
                pos_tripl[key] = []
            
            pos_tripl[key].append(folder_path + "/positive_pairs/" + im)

        #save combinations
        combis = []
        for k, ims in zip(pos_tripl.keys(), pos_tripl.values()):
            combis += combinations(ims,2)
        
        to_append = pd.DataFrame(combis, columns = ["gt_image_path","impainted_pth"])

        self.df = pd.concat([self.df,to_append])
        
        self.df = self.df.reset_index()
        self.model_processor = model_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        #found some images that are black and white, so we convert them to RGB
        img_a = Image.open(row['gt_image_path']).convert('RGB')
        img_b = Image.open(row['impainted_pth']).convert('RGB')
        if pd.isna(row["segmentation_map_path"]):
            # If no segmentation map is provided, create all false mask
            mask = Image.new('RGB', img_a.size, color=(0, 0, 0))
        else:
            mask = Image.open(row["segmentation_map_path"]).convert('RGB')
            
        img_a = self.model_processor(img_a)
        img_b = self.model_processor(img_b)
        mask = self.model_processor(mask)
        
        #get only one channel from mask
        mask = mask[0, :, :] / mask.max() # Assuming mask is a single channel image
        mask = 1.0 - mask
        cosmap = mask
        cosmap = cosmap.unsqueeze(0) #add channel dim

        return (img_a, img_b), cosmap