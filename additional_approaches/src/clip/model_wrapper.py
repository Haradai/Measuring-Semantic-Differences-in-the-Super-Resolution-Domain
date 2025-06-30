import torch 
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
from  src.my_splice.splice_module_l1 import splice_wrapper #tst

class CLIP_wrapper():
    def __init__(self, model_name, device):
        '''
        This class aims to be a wrapper to embed images and text in a 
        convenient and compatible way with the pytorch_grad_cam implementation.
        '''
        #Define model and preprocessors
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.img_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.txt_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.patch_grid_size = 17
        self.image_size = 224
        
        self.model_name = model_name.split("/")[1] #avoid openai/
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
            
        '''
        This wrapper classes are needed so that img_embedder model or img_embedder can be passed to the gradcam
        implementation directly as it will call the passed model and compute upon the output directly. 
        If we want to change the embeddings after clip this has to be done here.
        '''
        class text_embedder(torch.nn.Module):
            def __init__(self, model, tokenizer,device, patch_grid_size, image_size):
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, text):
                ''' 
                Directly pass text
                '''
                return self.model.get_text_features(**self.tokenizer(text ,return_tensors = "pt", padding = True).to(self.device))

        class img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                self.model = model
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size
                self.individual_tokens_mode = False
                self.depth_return = "d0" #depth is 0 last layer features and so on ONLY WORKS WITH INDIVIDUAL TOKENS MODE ON

            def forward(self, pixel_vals):
                ''' 
                Pass preprocessed image, use model.load_image
                '''
                if self.individual_tokens_mode:
                    
                    #code borrowed from the .get_image_features function definition
                    # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
                    output_attentions = self.model.config.output_attentions
                    output_hidden_states = (
                        self.model.config.output_hidden_states
                    )
                    return_dict = self.model.config.use_return_dict

                    vision_outputs = self.model.vision_model(
                            pixel_values=pixel_vals,
                            output_attentions=output_attentions,
                            output_hidden_states=True,
                            return_dict=return_dict,
                        )
                    

                    if self.depth_return == "d0":
                        #get last hidden layer tokens feats and project them into shared emb space
                        proj_tokens = self.model.visual_projection(vision_outputs["last_hidden_state"])
                        
                        return proj_tokens

                    elif self.depth_return == "all":
                        return vision_outputs

                    else:
                        raise ValueError("Depth string not recognized!")
                else:
                    return self.model.get_image_features(pixel_vals)
            
        #Wrapper model instances:
        self.img_embedder = img_embedder(self.model, self.load_img,self.patch_grid_size, self.image_size) 
        self.txt_embedder = text_embedder(self.model,self.txt_tokenizer,self.device,self.patch_grid_size, self.image_size)
        

    def load_img(self,img_pth) -> torch.tensor:
        if isinstance(img_pth,list):
            image = [Image.open(pth) for pth in img_pth]
        else:
            image = Image.open(img_pth)

        processed_img = self.img_processor(image,return_tensors="pt")
        return processed_img["pixel_values"].to(self.device).requires_grad_(True)
    

