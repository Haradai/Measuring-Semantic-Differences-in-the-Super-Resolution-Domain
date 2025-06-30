from ram import inference_ram as inference
from ram.models import ram_plus
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch


class ram_wrapper:
    def __init__(self,device):
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
        self.device = device

        self.model = ram_plus(pretrained="src/ram/ram_plus_swin_large_14m.pth", vit="swin_l", image_size=384).to(self.device)

    def __call__(self,img_pth:str):
        image = Image.open(img_pth)
        img_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        out = inference(img_tensor,self.model)

        return out[0].split(" | ") #english output


class ram_embedding_projector():
    def __init__(self,device):
        self.ram = ram_wrapper(device)
        self.ret_concepts = False

    def __call__(self,img_pth:str, img_embedder, txt_embedder) -> torch.Tensor:
        words = self.ram(img_pth)

        #embed found words
        concepts = txt_embedder(words).detach().cpu()

        #embed image
        img = img_embedder.load_img(img_pth)
        img_emb = img_embedder(img).detach().cpu()
        
        #normalize txt embs
        concepts /= torch.linalg.norm(concepts,dim=0)

        #dot products
        D = torch.matmul(concepts,img_emb.squeeze(0)) #dot products

        #normalize dot products so they add up to 1
        D /= D.sum()
        
        rec = torch.sum(concepts*D.reshape(-1,1),axis=0)

        if self.ret_concepts:
            return rec, words
        
        else:
            return rec
