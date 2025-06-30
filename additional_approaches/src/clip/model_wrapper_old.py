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
                return self.model.get_text_features(**self.tokenizer(text ,return_tensors = "pt", padding = True).to(self.device))

        class img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                self.model = model
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size
                self.individual_tokens_mode = False

            def forward(self, pixel_vals):
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
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                        )
                    
                    #get last hidden layer tokens feats and project them into shared emb space
                    proj_tokens = self.model.visual_projection(vision_outputs["last_hidden_state"])
                    
                    return proj_tokens
                
                else:
                    return self.model.get_image_features(pixel_vals)
        
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
                emb = self.model.get_image_features(pixel_vals)
                focus_emb = self.focus_emb / torch.norm(self.focus_emb) #normalize just in case the focus emb
                focus_emb = focus_emb.repeat(emb.shape[0],1)
                scalars = (emb * focus_emb).sum(dim=1).unsqueeze(1) #dot product of all embs with focus_emb
                focused_embs = focus_emb * scalars

                return focused_embs
        
        class unfocus_img_embedder(torch.nn.Module):
            def __init__(self, model,load_img,patch_grid_size, image_size):
                super().__init__()
                '''
                This wrapper should be used by changing the focus_emb attribute first and then running the model.
                e.g clip_wrapper.unfocus_img_embedder.unfocus_emb = <mytext embedding tensor>
                
                It projects the embedding onto focus_emb direction.
                '''
                self.model = model
                self.unfocus_emb = None
                self.loa_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, pixel_vals):
                emb = self.model.get_image_features(pixel_vals)
                focus_emb = self.unfocus_emb / torch.norm(self.unfocus_emb) #normalize just in case the focus emb
                focus_emb = focus_emb.repeat(emb.shape[0],1)
                scalars = (emb * focus_emb).sum(dim=1).unsqueeze(1) #dot product of all embs with focus_emb
                focused_embs = focus_emb * scalars
                
                return emb - focused_embs
        
        class splice_focus_img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size, model_name):
                super().__init__()
                '''
                This wrapper is used to embed images but focusing on concepts extracted by splice.
                You can input rank (number of words in the decomposition basis) and you can input also a basis vocabulary.
                If no basis vocabulary is passed it will just take the first (rank size) of the default laion dictionary.
                '''
                self.model = model
                self.load_img = load_img
                self.method = "skl" #can be changed to admm to have gradients
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size
                self.l1 = 3
                self.rank = 5
                self.target_mode = "rank" #can be rank or l1
                self.weights_mode = False

                #load splice module
                self.splice = splice_wrapper(
                    f"src/clip/{model_name}_laion_embeddings.pt",
                    f"src/clip/{model_name}_image_mean.pt",
                    self.model.device,
                    )


            def forward(self, pixel_vals):
                #embed img
                emb = self.model.get_image_features(pixel_vals)

                #if skl cannot do with grads
                if self.method == "skl":
                    emb = emb.detach()
                
                #decompose embeddings
                if self.target_mode == "l1":
                    weights = self.splice.decompose(emb, self.method, l1 = self.l1)
                
                elif self.target_mode == "rank":
                    weights = self.splice.decompose(emb, self.method, rank = self.rank)

                #could happen that for the given l1 or rank we were unable to decompose, in that case return none
                if weights is None:
                    return None
                
                #if weights mode then directly return that
                if self.weights_mode:
                    return weights
                
                #recompose weights into embedding
                rec_emb = self.splice.recompose(weights)
                
                return rec_emb
            
        #Wrapper model instances:
        self.img_embedder = img_embedder(self.model, self.load_img,self.patch_grid_size, self.image_size) 
        self.txt_embedder = text_embedder(self.model,self.txt_tokenizer,self.device,self.patch_grid_size, self.image_size)
        
        self.focus_img_embedder = focus_img_embedder(self.model, self.load_img,self.patch_grid_size, self.image_size) 
        self.unfocus_img_embedder = unfocus_img_embedder(self.model, self.load_img,self.patch_grid_size, self.image_size) 
        self.splice_focus_img_embedder = splice_focus_img_embedder(self.model, self.load_img, self.patch_grid_size, self.image_size, self.model_name) 

    def load_img(self,img_pth) -> torch.tensor:
        if isinstance(img_pth,list):
            image = [Image.open(pth) for pth in img_pth]
        else:
            image = Image.open(img_pth)

        processed_img = self.img_processor(image,return_tensors="pt")
        return processed_img["pixel_values"].to(self.device).requires_grad_(True)
    

