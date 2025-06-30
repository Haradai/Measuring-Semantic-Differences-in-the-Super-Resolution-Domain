import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytora import apply_lora

class CLIP_lpips_Unet(nn.Module):
    def __init__(self, clip_name:str, device:str, lora_rank:int=None):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_Unet, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.lora_rank = lora_rank
        
        if lora_rank != "full":
            self.clip.eval()
        
        if lora_rank != "full":
            if self.lora_rank is not None:
                ##Apply LORA to the CLIP model
                apply_lora(self.clip, lora_r = lora_rank)

        self.clip = self.clip.to(device)
        
        #Define clip hooks to get layer hidden features
        self.wanted_layers = ["stem.conv3"] + [f"stages.{s}.{2}.act" for s in range(4)] 
        
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        #DECODER
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256+64, 64, kernel_size=3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1, padding='same'),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(256+512, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(512+1024, 512, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),

             nn.Sequential(
                nn.Conv2d(1024+2048, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            ),
            
            nn.Sequential(
                nn.Conv2d(2048, 2048, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, kernel_size=3, padding='same'),
                nn.BatchNorm2d(2048),
                nn.ReLU()
            )
        ])

        self.upscaler = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.final_sigmoid = torch.nn.Sigmoid()
        self.init_weights() #intiialize weights

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save the hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save the hidden features

        
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())

        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]

        
        #DECODE
        bottom_pass = self.decoder[-1](diff[-1])
        bottom_pass = self.upscaler(bottom_pass)
        for j in range(2,len(diff)+1):
            to_pass = torch.concat( (diff[-j], bottom_pass), dim=1) #concat along channel dim
            bottom_pass = self.decoder[-j](to_pass)
            bottom_pass = self.upscaler(bottom_pass)

        bottom_pass =  self.final_sigmoid(bottom_pass)
        return bottom_pass

    def _register_hooks(self):
        """Register forward hooks on the specified layers."""
        for name, module in self.clip.named_modules():
            if name in self.wanted_layers:
                module.register_forward_hook(self._get_hook(name))
    
    def _get_hook(self, name: str):
        """
        Create a hook function to capture the output of a specific layer.
        
        Args:
            name: The name of the layer
            
        Returns:
            A hook function
        """
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def init_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def save_model(self,path:str):
        if self.lora_rank is not None:
            # Save the model with LoRA weights
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.decoder.state_dict(), path)

    def load_model(self,path:str):
        if self.lora_rank is not None:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.decoder.load_state_dict(torch.load(path, weights_only=True))
       


class CLIP_lpips_Unet_clsbckbn(nn.Module):
    def __init__(self, clip_name:str, device:str, lora_rank:int=None):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_Unet_clsbckbn, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.lora_rank = lora_rank
        
        if lora_rank != "full":
            self.clip.eval()
        
        if lora_rank != "full":
            if self.lora_rank is not None:
                ##Apply LORA to the CLIP model
                apply_lora(self.clip, lora_r = lora_rank)

        self.clip = self.clip.to(device)
        
        #Define clip hooks to get layer hidden features
        self.wanted_layers = ["conv1"] + [f"layer{s}.2.act3" for s in range(1,5)] 
        
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        #DECODER
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256+64, 64, kernel_size=3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1, padding='same'),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(256+512, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(512+1024, 512, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),

             nn.Sequential(
                nn.Conv2d(1024+2048, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            ),
            
            nn.Sequential(
                nn.Conv2d(2048, 2048, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, kernel_size=3, padding='same'),
                nn.BatchNorm2d(2048),
                nn.ReLU()
            )
        ])

        self.upscaler = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.final_sigmoid = torch.nn.Sigmoid()
        self.init_weights() #intiialize weights

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save the hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save the hidden features

        
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())

        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]

        
        #DECODE
        bottom_pass = self.decoder[-1](diff[-1])
        bottom_pass = self.upscaler(bottom_pass)
        for j in range(2,len(diff)+1):
            to_pass = torch.concat( (diff[-j], bottom_pass), dim=1) #concat along channel dim
            bottom_pass = self.decoder[-j](to_pass)
            bottom_pass = self.upscaler(bottom_pass)

        bottom_pass =  self.final_sigmoid(bottom_pass)
        return bottom_pass

    def _register_hooks(self):
        """Register forward hooks on the specified layers."""
        for name, module in self.clip.named_modules():
            if name in self.wanted_layers:
                module.register_forward_hook(self._get_hook(name))
    
    def _get_hook(self, name: str):
        """
        Create a hook function to capture the output of a specific layer.
        
        Args:
            name: The name of the layer
            
        Returns:
            A hook function
        """
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def init_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def save_model(self,path:str):
        if self.lora_rank is not None:
            # Save the model with LoRA weights
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.decoder.state_dict(), path)

    def load_model(self,path:str):
        if self.lora_rank is not None:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.decoder.load_state_dict(torch.load(path, weights_only=True))



class CLIP_lpips_Unet_v2(nn.Module):
    def __init__(self, clip_name:str, device:str, lora_rank:int=None):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_Unet_v2, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.lora_rank = lora_rank
        
        if lora_rank != "full":
            self.clip.eval()
        
        if lora_rank != "full":
            if self.lora_rank is not None:
                ##Apply LORA to the CLIP model
                apply_lora(self.clip, lora_r = lora_rank)

        self.clip = self.clip.to(device)
        
        #Define clip hooks to get layer hidden features
        self.wanted_layers = ["stem.conv3"] + [f"stages.{s}.{2}.act" for s in range(4)] 
        
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        #DECODER
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256+64+1, 64, kernel_size=3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1, padding='same'),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(256+512+1, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(512+1024+1, 512, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),

             nn.Sequential(
                nn.Conv2d(1024+2048+1, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            ),
            
            nn.Sequential(
                nn.Conv2d(2048+1, 2048, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, kernel_size=3, padding='same'),
                nn.BatchNorm2d(2048),
                nn.ReLU()
            )
        ])

        self.upscaler = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.final_sigmoid = torch.nn.Sigmoid()
        self.init_weights() #intiialize weights

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save the hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save the hidden features

        img_squared = torch.mean((a - b) ** 2, dim=1, keepdim=True)
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())

        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]
        #add as extra channel img_squared to all features 
        diff_with_img = []
        for f in diff:
            img_resized = F.interpolate(img_squared, size=f.shape[2:], mode='bilinear', align_corners=False)
            diff_with_img.append(torch.cat([f, img_resized], dim=1))
        
        diff = diff_with_img

        
        #DECODE
        bottom_pass = self.decoder[-1](diff[-1])
        bottom_pass = self.upscaler(bottom_pass)
        for j in range(2,len(diff)+1):
            to_pass = torch.concat( (diff[-j], bottom_pass), dim=1) #concat along channel dim
            bottom_pass = self.decoder[-j](to_pass)
            bottom_pass = self.upscaler(bottom_pass)

        bottom_pass =  self.final_sigmoid(bottom_pass)
        return bottom_pass

    def _register_hooks(self):
        """Register forward hooks on the specified layers."""
        for name, module in self.clip.named_modules():
            if name in self.wanted_layers:
                module.register_forward_hook(self._get_hook(name))
    
    def _get_hook(self, name: str):
        """
        Create a hook function to capture the output of a specific layer.
        
        Args:
            name: The name of the layer
            
        Returns:
            A hook function
        """
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def init_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def save_model(self,path:str):
        if self.lora_rank is not None:
            # Save the model with LoRA weights
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.decoder.state_dict(), path)

    def load_model(self,path:str):
        if self.lora_rank is not None:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.decoder.load_state_dict(torch.load(path, weights_only=True))
       


class CLIP_lpips_Unet_clsbckbn_v2(nn.Module):
    def __init__(self, clip_name:str, device:str, lora_rank:int=None):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_Unet_clsbckbn_v2, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.lora_rank = lora_rank
        
        if lora_rank != "full":
            self.clip.eval()
        
        if lora_rank != "full":
            if self.lora_rank is not None:
                ##Apply LORA to the CLIP model
                apply_lora(self.clip, lora_r = lora_rank)

        self.clip = self.clip.to(device)
        
        #Define clip hooks to get layer hidden features
        self.wanted_layers = ["conv1"] + [f"layer{s}.2.act3" for s in range(1,5)] 
        
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        #DECODER
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256+64+1, 64, kernel_size=3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1, padding='same'),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(256+512+1, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(512+1024+1, 512, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),

             nn.Sequential(
                nn.Conv2d(1024+2048+1, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            ),
            
            nn.Sequential(
                nn.Conv2d(2048+1, 2048, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, kernel_size=3, padding='same'),
                nn.BatchNorm2d(2048),
                nn.ReLU()
            )
        ])

        self.upscaler = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.final_sigmoid = torch.nn.Sigmoid()
        self.init_weights() #intiialize weights

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save the hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save the hidden features

        
        img_squared = torch.mean((a - b) ** 2, dim=1, keepdim=True)
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())
glo
        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]
        #add as extra channel img_squared to all features 
        diff_with_img = []
        for f in diff:
            img_resized = F.interpolate(img_squared, size=f.shape[2:], mode='bilinear', align_corners=False)
            diff_with_img.append(torch.cat([f, img_resized], dim=1))
        
        diff = diff_with_img
        
        #DECODE
        bottom_pass = self.decoder[-1](diff[-1])
        bottom_pass = self.upscaler(bottom_pass)
        for j in range(2,len(diff)+1):
            to_pass = torch.concat( (diff[-j], bottom_pass), dim=1) #concat along channel dim
            bottom_pass = self.decoder[-j](to_pass)
            bottom_pass = self.upscaler(bottom_pass)

        bottom_pass =  self.final_sigmoid(bottom_pass)
        return bottom_pass

    def _register_hooks(self):
        """Register forward hooks on the specified layers."""
        for name, module in self.clip.named_modules():
            if name in self.wanted_layers:
                module.register_forward_hook(self._get_hook(name))
    
    def _get_hook(self, name: str):
        """
        Create a hook function to capture the output of a specific layer.
        
        Args:
            name: The name of the layer
            
        Returns:
            A hook function
        """
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def init_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def save_model(self,path:str):
        if self.lora_rank is not None:
            # Save the model with LoRA weights
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.decoder.state_dict(), path)

    def load_model(self,path:str):
        if self.lora_rank is not None:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.decoder.load_state_dict(torch.load(path, weights_only=True))