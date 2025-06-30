import torch 
from PIL import Image
from transformers import BertTokenizer
from src.albef.vit import interpolate_pos_embed
from src.albef.model_retrieval import ALBEF
import torch.nn.functional as F
from torchvision import transforms
import yaml
from pathlib import Path

class ALBEF_wrapper():
    def __init__(self, checkpoint:str="retrieval", device="cuda"):
        '''
        This class aims to be a wrapper to embed images and text in a 
        convenient and compatible way with the pytorch_grad_cam implementation.
        '''
        
        #checkpoint files
        if checkpoint == "retrieval":
            config_pth = "src/albef/configs/Retrieval_flickr.yaml"
            checkpint_pth = "src/albef/flickr30k.pth"

        elif checkpoint == "grounding":
            config_pth = "src/albef/configs/grounding.yaml"
            checkpint_pth = "src/albef/refcoco.pth"
        else:
            raise ValueError("checkpoint name not valid!")

        #Define model and preprocessors
        self.device = device
        self.txt_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.patch_grid_size = 24
        self.image_size = 384
        self.name = checkpoint

        #load image mean
        self.image_mean = torch.load(f"src/albef/{checkpoint}_image_mean.pt")
        
        #save here path for laion dataset embeddings pt path
        self.laion_embs_pth = f"src/albef/{checkpoint}_laion_embeddings.pt"

        conf = yaml.safe_load(Path(config_pth).read_text())
        self.model =  ALBEF(config=conf, text_encoder="bert-base-uncased", tokenizer=self.txt_tokenizer)
        
        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        #Load checkpoint 
        state_dict = torch.load(checkpint_pth, map_location='cpu') 

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],self.model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]       

        msg = self.model.load_state_dict(state_dict,strict=False)  
        
        self.model = self.model.to(device)
        print(f'load checkpoint from {checkpint_pth}')
        print(msg)  
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
            
        '''
        This wrapper classes are needed so that img_embedder model or img_embedder can be passed to the gradcam
        implementation directly as it will call the passed model and compute upon the output directly. 
        If we want to change the embeddings after clip this has to be done here.
        '''
        class text_embedder(torch.nn.Module):
            def __init__(self, model, tokenizer, device, patch_grid_size, image_size):
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size
                
            def forward(self, text):
                text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(self.device)
                text_output = self.model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
                text_feat = text_output.last_hidden_state
                return F.normalize(self.model.text_proj(text_feat[:,0,:]))

        class img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                self.model = model
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, pixel_vals):
                image_feat = self.model.visual_encoder(pixel_vals)        
                image_embed = self.model.vision_proj(image_feat[:,0,:])            
                return F.normalize(image_embed,dim=-1)      
        
        class focus_img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                '''
                This wrapper should be used by changing the focus_emb attribute first and then running the model.
                e.g clip_wrapper.focus_img_embedder.focus_emb = <mytext embedding tensor>
                
                It projects the embedding onto focus_emb direction.
                '''
                self.model = model
                self.focus_emb = None
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, pixel_vals):
                #embed img
                image_feat = self.model.visual_encoder(pixel_vals)        
                image_embed = self.model.vision_proj(image_feat[:,0,:])
                emb = F.normalize(image_embed,dim=-1)      
                
                #normalize focus emb
                focus_emb = self.focus_emb / torch.norm(self.focus_emb)
                focus_emb = focus_emb.repeat(emb.shape[0],1)
                scalars = (emb * focus_emb).sum(dim=1).unsqueeze(1) #dot product of all embs with focus_emb
                focused_embs = focus_emb * scalars

                return focused_embs
        
        class unfocus_img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                '''
                This wrapper should be used by changing the focus_emb attribute first and then running the model.
                e.g clip_wrapper.focus_img_embedder.focus_emb = <mytext embedding tensor>
                
                It projects the embedding onto focus_emb direction.
                '''
                self.model = model
                self.unfocus_emb = None
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size
                
            def forward(self, pixel_vals):
                #embed img
                image_feat = self.model.visual_encoder(pixel_vals)        
                image_embed = self.model.vision_proj(image_feat[:,0,:])
                emb = F.normalize(image_embed,dim=-1)      
                
                #normalize focus emb
                unfocus_emb = self.unfocus_emb / torch.norm(self.unfocus_emb)
                unfocus_emb = unfocus_emb.repeat(emb.shape[0],1)
                scalars = (emb * unfocus_emb).sum(dim=1).unsqueeze(1) #dot product of all embs with focus_emb
                focused_embs = unfocus_emb * scalars

                return emb - focused_embs #remove projected component

        #Wrapper model instances:
        self.img_embedder = img_embedder(self.model, self.load_img, self.patch_grid_size, self.image_size) 
        self.txt_embedder = text_embedder(self.model,self.txt_tokenizer,self.device,self.patch_grid_size, self.image_size)
        self.focus_img_embedder = focus_img_embedder(self.model, self.load_img,self.patch_grid_size, self.image_size) 
        self.unfocus_img_embedder = unfocus_img_embedder(self.model, self.load_img,self.patch_grid_size, self.image_size) 

    '''
    def load_img(self,img_pth:str) -> torch.tensor:
        image = Image.open(img_pth)
        processed_img = self.image_transform(image).unsqueeze(0).to(self.device)
        return  processed_img
    '''
    def load_img(self, img_paths: list[str]) -> torch.Tensor:
        
        if isinstance(img_paths,list):
            # Open all images
            images = [Image.open(img_path) for img_path in img_paths]
            
            # Apply transformations to all images and convert to tensor
            processed_imgs = torch.stack([
                self.image_transform(image) for image in images
            ]).to(self.device)
            
            return processed_imgs
        
        else:
            image = Image.open(img_paths)
            processed_img = self.image_transform(image).unsqueeze(0).to(self.device)
            return  processed_img
