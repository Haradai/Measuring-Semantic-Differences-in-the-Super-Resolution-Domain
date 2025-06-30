from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
from caption_generation_vipllava import panoptic_to_masks
from tqdm import tqdm 
import pandas as pd
from itertools import combinations
from tqdm.notebook import tqdm

def load_emb_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

def load_captions(pth:str):
    """pip install flash-attn
    Load the captions from the Koniq-10k dataset.
    Returns:
        dict: A dictionary mapping image names to their captions.
    """
    with open(pth, "r") as file:
        hq_captions = file.read()

    #split by image 
    hq_captions = hq_captions.split("\n\nNEXT_CAPTION\n\n")

    #split every into image number and caption
    hq_captions = [caption.split("Image: ") for caption in hq_captions]

    #flatten the list
    hq_captions = [item for sublist in hq_captions for item in sublist]

    #filter actual lines
    hq_captions = [c for c in hq_captions if c != "" and c != "\n"]

    #remove last \n
    hq_captions = [c[:] for c in hq_captions]

    hq_captions = {a.split("\nCaption:")[0].split(".")[0] : a.split("\nCaption:")[1] for a in hq_captions}
    
    
    return hq_captions
    
def merge_caption_dicts(dicts, folder_paths) -> dict:
    merged = {}
    
    #We'll save the keys as the path of the image each caption belongs to
    for d, f in zip(dicts, folder_paths):
        #find image extension of folder
        ext = os.listdir(f)
        ext = [e for e in ext if e.endswith((".jpg", ".png"))]
        ext = ext[0].split(".")[-1]

        for key, cap in d.items():
            merged[f"{f}/{key}.{ext}"] = cap
    
    return merged

###COMPUTE 
folders = [
    "koniq-10k_sr/HQ",
    "koniq-10k_sr/LQ_x4_degfac_0.7_bsrgan_out",
    "koniq-10k_sr/LQ_x4_degfac_0.7_swinir_out",
    "koniq-10k_sr/LQ_x4_degfac_0.7_seesr_out",
    "koniq-10k_sr/LQ_x4_degfac_0.7_pasd_out",
    "koniq-10k_sr/LQ_x4_degfac_0.7_stablesr_out",
    "koniq-10k_sr/LQ_x4_degfac_0.3_bsrgan_out",
    "koniq-10k_sr/LQ_x4_degfac_0.3_swinir_out",
    "koniq-10k_sr/LQ_x4_degfac_0.3_seesr_out",
    "koniq-10k_sr/LQ_x4_degfac_0.3_pasd_out",
    "koniq-10k_sr/LQ_x4_degfac_0.3_stablesr_out",
    "koniq-10k_sr/LQ_x4_bsrgan_out",
    "koniq-10k_sr/LQ_x4_swinir_out",
    "koniq-10k_sr/LQ_x4_seesr_out",
    "koniq-10k_sr/LQ_x4_pasd_out",
    "koniq-10k_sr/LQ_x4_stablesr_out"
]


#load all captions of all images

all_cap_dicts = []
for f in folders:
    output_dir = os.path.join(f, "whole_im_embs")
    os.makedirs(output_dir, exist_ok=True)

    caps_pth = f + "/wholeim_captions_batch.txt"
    captions = load_captions(f + "/wholeim_captions_batch.txt") #load captions
    all_cap_dicts.append(captions) #merge caps

#merge all captions into one dict
captions = merge_caption_dicts(all_cap_dicts, folders)


##NOW COMPUTE ALL POSSIBLE PAIRS OF IMAGES
hq_ims = []
sr_ims = []

for f in folders:
    ims = os.listdir(f)
    ims = [i for i in ims if i.endswith('.jpg') or i.endswith('.png')]
    hq_ims.extend(["koniq-10k_sr/HQ/" + e.split(".")[0] + ".jpg" for e in ims])
    sr_ims.extend([f"{f}/" + e for e in ims])

pairs = list(zip(hq_ims,sr_ims))
df = pd.DataFrame(pairs, columns=["hq","sr"])

#make between sr images pairs
for hq in df["hq"].unique():
    subset = list(df[df["hq"] ==  hq]["sr"])
    combis = combinations(subset,2)
    pairs.extend(combis)

df = pd.DataFrame(pairs, columns=["img_a_pth","img_b_pth"])

#LOAD EMBEDDING MODEL
model = load_emb_model()

#compute cosine similarity for each pair
def compute_cosine_sim(ptha:str, pthb:str) -> float:
    #find captions and embed
    caps = [captions[ptha], captions[pthb]]
    
    #embed
    embeddings = model.encode(caps)

    #compute cosine similarity
    cos = np.dot(embeddings[0,:], embeddings[1,:])/(np.linalg.norm(embeddings[0,:])*np.linalg.norm(embeddings[1,:]))

    return cos

cosines = []
for j, row in df.iterrows():
    if j % 100 == 0:
        print(j)
    cosines.append(compute_cosine_sim(row["img_a_pth"], row["img_b_pth"]))

#add cosines to df
df["cosine"] = cosines

#save to csv
df.to_csv("koniq-10k_sr/wholeim_pairs_cosines.csv", index=False)