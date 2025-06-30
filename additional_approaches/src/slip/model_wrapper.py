import torch 
from PIL import Image
from collections import OrderedDict
from src.slip.models import SLIP_VITB16
from src.slip.tokenizer import SimpleTokenizer
import torchvision.transforms as transforms

from  src.my_splice.splice_module_l1 import splice_wrapper #tst

class SLIP_wrapper():
    def __init__(self, device):
        '''
        This class aims to be a wrapper to embed images and text in a 
        convenient and compatible way with the pytorch_grad_cam implementation.
        '''
        #Define model and preprocessors
        self.device = device

        #load checkpoint
        ckpt = torch.load("src/slip/slip_base_100ep.pt", map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        #load model
        self.model = SLIP_VITB16(ssl_mlp_dim=4096,ssl_emb_dim=256, rand_embed=False)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)

        self.txt_tokenizer = SimpleTokenizer("src/slip/bpe_simple_vocab_16e6.txt.gz")

        self.img_processor = transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                lambda x: x.convert('RGB'),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                    ])
        self.patch_grid_size = 17
        self.image_size = 224
        
        self.model_name = "slip_base_100ep"
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
            
        '''
        This wrapper classes are needed so that img_embedder model or img_embedder can be passed to the gradcam
        implementation directly as it will call the passed model and compute upon the output directly. 

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
                return self.model.encode_text(self.tokenizer(text).to(self.device))

        class img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                self.model = model
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, pixel_vals):
                return self.model.encode_image(pixel_vals)
        
        
        class focus_img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size):
                super().__init__()
                
                #This wrapper should be used by changing the focus_emb attribute first and then running the model.
                #e.g clip_wrapper.focus_img_embedder.focus_emb = <mytext embedding tensor>
                #It projects the embedding onto focus_emb direction.
                
                self.model = model
                self.focus_emb = None
                self.load_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, pixel_vals):
                emb =  self.model.encode_image(pixel_vals)
                focus_emb = self.focus_emb / torch.norm(self.focus_emb) #normalize just in case the focus emb
                focus_emb = focus_emb.repeat(emb.shape[0],1)
                scalars = (emb * focus_emb).sum(dim=1).unsqueeze(1) #dot product of all embs with focus_emb
                focused_embs = focus_emb * scalars

                return focused_embs
        
        class unfocus_img_embedder(torch.nn.Module):
            def __init__(self, model,load_img,patch_grid_size, image_size):
                super().__init__()
                
                #This wrapper should be used by changing the focus_emb attribute first and then running the model.
                #e.g clip_wrapper.unfocus_img_embedder.unfocus_emb = <mytext embedding tensor>
                
                #It projects the embedding onto focus_emb direction.
                
                self.model = model
                self.unfocus_emb = None
                self.loa_img = load_img
                self.patch_grid_size = patch_grid_size
                self.image_size = image_size

            def forward(self, pixel_vals):
                emb = self.model.encode_image(pixel_vals)
                focus_emb = self.unfocus_emb / torch.norm(self.unfocus_emb) #normalize just in case the focus emb
                focus_emb = focus_emb.repeat(emb.shape[0],1)
                scalars = (emb * focus_emb).sum(dim=1).unsqueeze(1) #dot product of all embs with focus_emb
                focused_embs = focus_emb * scalars
                
                return emb - focused_embs
        
        class splice_focus_img_embedder(torch.nn.Module):
            def __init__(self, model, load_img, patch_grid_size, image_size, model_name):
                super().__init__()
                
                #This wrapper is used to embed images but focusing on concepts extracted by splice.
                #You can input rank (number of words in the decomposition basis) and you can input also a basis vocabulary.
                #If no basis vocabulary is passed it will just take the first (rank size) of the default laion dictionary.
                
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
                    f"src/slip/{model_name}_laion_embeddings.pt",
                    f"src/slip/{model_name}_image_mean.pt",
                    self.model.device,
                    )


            def forward(self, pixel_vals):
                #embed img
                emb = self.model.encode_image(pixel_vals)

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
        #self.splice_focus_img_embedder = splice_focus_img_embedder(self.model, self.load_img, self.patch_grid_size, self.image_size, self.model_name) 

    def load_img(self,img_pth) -> torch.tensor:
        if isinstance(img_pth,list):
            image = [self.img_processor(Image.open(pth)) for pth in img_pth]
            processed_img = torch.stack(image)
        else:
            image = self.img_processor(Image.open(img_pth))
            processed_img = image
        
        return processed_img.to(self.device).requires_grad_(True)
    

