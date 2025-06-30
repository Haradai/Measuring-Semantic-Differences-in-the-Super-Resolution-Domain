import torch
from torch import nn
import timm
import numpy as np

class CLIP_lpips_singleLin_vit(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_singleLin_vit, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.clip.eval()
        self.clip.to(device)


        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"blocks.{l}.ls2" for l in range(11-depth,11+1)]

        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        #define attention w module ASSUMING HERE IS ONLY ONE FC + ACT
        self.w_layer = nn.Sequential(
            nn.Linear(768,1),
            )
        
        #final relu
        self.final_relu = nn.ReLU()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = torch.stack(list(a_hidden_feats.values()))
        feats_b = torch.stack(list(b_hidden_feats.values()))


        #change layer dim for batch dim

        feats_a = feats_a.swapaxes(0,1)
        feats_b = feats_b.swapaxes(0,1)
    

        diff =  (feats_a - feats_b) ** 2
        
        ws = self.w_layer(diff).squeeze(-1) #remove last dim

        #spatial average, along tokens
        s_feats = torch.mean(ws,dim=-1) 

        #average along layers
        s_feats = torch.mean(s_feats,dim=-1) 

        #pass through ReLU to help
        s_feats = self.final_relu(s_feats)

        return s_feats

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
    

class CLIP_lpips_stages_vit(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_stages_vit, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.clip.eval()
        self.clip.to(device)
        self.depth = depth 

        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"blocks.{l}.ls2" for l in range(11-(depth * 3),11+1, 3)]  #we get groups of three every three (this is to be consistent with resnet variant implementation 4 blocks of 3)
        print(self.wanted_layers)
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        self.w_layers = nn.ModuleList([nn.Linear(768, 1) for _ in range(depth + 1)])
        
        #final relu
        self.final_relu = nn.ReLU()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = torch.stack(list(a_hidden_feats.values()))
        feats_b = torch.stack(list(b_hidden_feats.values()))

        #change layer dim for batch dim

        feats_a = feats_a.swapaxes(0,1)
        feats_b = feats_b.swapaxes(0,1)
    
        diff =  (feats_a - feats_b) ** 2
        
        ws = []
        for j in range(self.depth + 1):
            ws.append(self.w_layers[j](diff[:,j,:,:]).squeeze(-1)) #also remove extra ch dim 1
        
        if len(ws) == 1:
            s_feats = ws[0]

        else:
            ws = torch.stack(ws)
            ws = ws.swapaxes(0,1)#change layer dim for batch dim
           
            #average along layers
            s_feats = torch.mean(ws,dim=-1) 
            
        #spatial average, along tokens
        s_feats = torch.mean(s_feats,dim=-1) 

        print(s_feats.shape)
        #pass through ReLU to help
        s_feats = self.final_relu(s_feats)
        
        return s_feats

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


class CLIP_lpips_wperlay_vit(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_wperlay_vit, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.clip.eval()
        self.clip.to(device)
        self.depth = depth 

        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"blocks.{l}.ls2" for l in range(11-depth,11+1)]  #we get groups of three every three (this is to be consistent with resnet variant implementation 4 blocks of 3)

        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        self.w_layers = nn.ModuleList([nn.Linear(768, 1) for _ in range(depth + 1)])
        
        #final relu
        self.final_relu = nn.ReLU()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = torch.stack(list(a_hidden_feats.values()))
        feats_b = torch.stack(list(b_hidden_feats.values()))

        #change layer dim for batch dim

        feats_a = feats_a.swapaxes(0,1)
        feats_b = feats_b.swapaxes(0,1)
    
        diff =  (feats_a - feats_b) ** 2
        
        ws = []
        for j in range(self.depth + 1):
            ws.append(self.w_layers[j](diff[:,j,:,:]).squeeze(-1))
        
        if len(ws) == 1:
            s_feats = ws[0]
        
        else:
            ws = torch.stack(ws)
            ws = ws.swapaxes(0,1)#change layer dim for batch dim
             
            #average along layers
            s_feats = torch.mean(ws,dim=-1) 
    

        #spatial average, along tokens
        s_feats = torch.mean(s_feats,dim=-1) 


        #pass through ReLU to help
        s_feats = self.final_relu(s_feats)

        return s_feats

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
    

class CLIP_lpips_stages_cnn(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str, enc_ft:bool=False):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_stages_cnn, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.enc_ft = enc_ft

        if not self.enc_ft:
            self.clip.eval()
        else:
            self.clip.train()

        self.clip.to(device)
        self.depth = depth 

        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"stages.{s}.{2}.act" for s in range(3-depth,4)] 
        print(self.wanted_layers)
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        self.w_layers = nn.ModuleList([nn.Conv2d(256 * (2**s), 1, kernel_size=1, stride=1) for s in range(3-depth,4)])
        
        #final relu
        self.final_relu = nn.ReLU()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        if self.enc_ft:
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features
        else:
            with torch.no_grad():
                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_a = self.clip(a)
                a_hidden_feats = self.outputs.copy() #save a hidden features


                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_b = self.clip(b)
                b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())

        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]
        
        ws = [self.w_layers[j](feat).squeeze(1) for j, feat in enumerate(diff)]  #remove remaining channel dim
        
        #spatial average
        s_feats = [torch.mean( torch.mean(feat,dim=-1), dim=-1) for feat in ws]
        if len(s_feats) == 1:
            s_feats = s_feats[0]
            
        else:
            #concatenate layers into a tensor
            s_feats = torch.stack(s_feats)
            #now first dim is layer dimensions, take mean along there also to have (batch,1)
            s_feats = torch.mean(s_feats,dim=0) 

        #pass through ReLU to help
        s_feats = self.final_relu(s_feats)

        return s_feats

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

    def save_model(self,path:str):
        if self.enc_ft:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.w_layers.state_dict(), path)
        
    def load_model(self,path:str):   
        if self.enc_ft:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.w_layers.load_state_dict(torch.load(path, weights_only=True))

class CLIP_lpips_stages_cnn_pooling(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str, enc_ft:bool=False):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_stages_cnn_pooling, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.enc_ft = enc_ft

        if not self.enc_ft:
            self.clip.eval()
        else:
            self.clip.train()

        self.clip.to(device)
        self.depth = depth 

        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"stages.{s}.{2}.act" for s in range(3-depth,4)] 
    
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)
        
        init_dim = np.sum([256 * (2**s) * 2 for s in range(3-depth,4)])
        self.fin_lin = nn.Sequential(
            nn.Linear(init_dim, 2056),
            nn.ReLU(),
            nn.Linear(2056, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )
        
        #final relu
        self.final_relu = nn.ReLU()
        
        self.init_weights()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        if self.enc_ft:
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features
        else:
            with torch.no_grad():
                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_a = self.clip(a)
                a_hidden_feats = self.outputs.copy() #save a hidden features


                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_b = self.clip(b)
                b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())
        
        #spatial average each fature
        feats_a = [torch.mean(a, dim=(-1,-2)) for a in feats_a]
        feats_b = [torch.mean(b, dim=(-1,-2)) for b in feats_b]

        #concat all channelwise 
        feats_a = torch.concat(feats_a,dim=1)
        feats_b = torch.concat(feats_b,dim=1)

        #concatenate the channels from A and B
        feats = torch.concat((feats_a,feats_b),dim=1)

        return self.fin_lin(feats).squeeze(-1)
    
    def init_weights(self):
        for m in self.fin_lin:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def save_model(self,path:str):
        if self.enc_ft:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.w_layers.state_dict(), path)
        
    def load_model(self,path:str):   
        if self.enc_ft:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.w_layers.load_state_dict(torch.load(path, weights_only=True))

class CLIP_lpips_stages_emb_lin(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str, enc_ft:bool=False):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_stages_emb_lin, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.enc_ft = enc_ft

        if not self.enc_ft:
            self.clip.eval()
        else:
            self.clip.train()

        self.clip.to(device)
        self.depth = depth 

        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"stages.{s}.{2}.act" for s in range(3-depth,4)] 
        print(self.wanted_layers)
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        self.fin_lin = nn.Sequential(
            nn.Linear(2048, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )
        
        self.init_weights()
        
    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        if self.enc_ft:
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features
        else:
            with torch.no_grad():
                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_a = self.clip(a)
                a_hidden_feats = self.outputs.copy() #save a hidden features


                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_b = self.clip(b)
                b_hidden_feats = self.outputs.copy() #save a hidden features

        
        ws = torch.concat((embedding_a,embedding_b),dim=1)

        return self.fin_lin(ws).squeeze(-1)
    
    def init_weights(self):
        for m in self.fin_lin:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def save_model(self,path:str):
        if self.enc_ft:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.w_layers.state_dict(), path)
        
    def load_model(self,path:str):   
        if self.enc_ft:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.w_layers.load_state_dict(torch.load(path, weights_only=True))

class CLIP_lpips_stages_cnn_clsbckb(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str, enc_ft:bool=False):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_stages_cnn_clsbckb, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.enc_ft = enc_ft

        if not self.enc_ft:
            self.clip.eval()
        else:
            self.clip.train()

        self.clip.to(device)
        self.depth = depth 

        #Define clip hooks to get layer hidden features
        self.wanted_layers = [f"layer{s}.2.act3" for s in range(4-depth,5)]
        print(self.wanted_layers)
        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        self.w_layers = nn.ModuleList([nn.Conv2d(256 * (2**s), 1, kernel_size=1, stride=1) for s in range(3-depth,4)])
        
        #final relu
        self.final_relu = nn.ReLU()

        self.init_weights()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        if self.enc_ft:
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features
        else:
            with torch.no_grad():
                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_a = self.clip(a)
                a_hidden_feats = self.outputs.copy() #save a hidden features


                # Clear previous outputs
                self.outputs = {}
                #forward pass a 
                embedding_b = self.clip(b)
                b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())

        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]
        
        ws = [self.w_layers[j](feat).squeeze(1) for j, feat in enumerate(diff)]  #remove remaining channel dim
        
        #spatial average
        s_feats = [torch.mean( torch.mean(feat,dim=-1), dim=-1) for feat in ws]
        if len(s_feats) == 1:
            s_feats = s_feats[0]
            
        else:
            #concatenate layers into a tensor
            s_feats = torch.stack(s_feats)
            #now first dim is layer dimensions, take mean along there also to have (batch,1)
            s_feats = torch.mean(s_feats,dim=0) 

        #pass through ReLU to help
        s_feats = self.final_relu(s_feats)

        return s_feats

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

    def save_model(self,path:str):
        if self.enc_ft:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.w_layers.state_dict(), path)
        
    def load_model(self,path:str):   
        if self.enc_ft:
            self.load_state_dict(torch.load(path, weights_only=True))
        else:
            self.w_layers.load_state_dict(torch.load(path, weights_only=True))

    def init_weights(self):
        for m in self.w_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

class CLIP_lpips_wperlay_cnn(nn.Module):
    def __init__(self, clip_name:str, depth:int, device:str, enc_ft:bool=False):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_wperlay_cnn, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.enc_ft = enc_ft
        if not self.enc_ft:
            self.clip.eval()
        
        self.clip.to(device)
        self.depth = depth 


        #Define clip hooks to get layer hidden features
        wanted_layers = [f"stages.{s}.{lay}.act" for s in range(4) for lay in range(3)] 
        self.wanted_layers = wanted_layers[11-depth:]

        #apply hooks to clip model 
        self._register_hooks()

        #clip preprocessor
        data_config = timm.data.resolve_model_data_config(self.clip)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        w_layers = []
        for lay in self.wanted_layers:
            stage = int(lay.split(".")[1])
            w_layers.append(nn.Conv2d(256 * (2**stage), 1, kernel_size=1, stride=1))

        self.w_layers = nn.ModuleList(w_layers)
        
        #final relu
        self.final_relu = nn.ReLU()

    def forward(self, a, b):
        '''
        '''
        #First get hidden features from clip

        with torch.no_grad():
            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_a = self.clip(a)
            a_hidden_feats = self.outputs.copy() #save a hidden features


            # Clear previous outputs
            self.outputs = {}
            #forward pass a 
            embedding_b = self.clip(b)
            b_hidden_feats = self.outputs.copy() #save a hidden features

        
        #stack features
        feats_a = list(a_hidden_feats.values())
        feats_b = list(b_hidden_feats.values())

        diff =  [(a-b) ** 2 for a,b in zip(feats_a, feats_b)]
        
        ws = [self.w_layers[j](feat).squeeze(1) for j, feat in enumerate(diff)]  #remove remaining channel dim
        
        #spatial average
        s_feats = [torch.mean( torch.mean(feat,dim=-1), dim=-1) for feat in ws]
        if len(s_feats) == 1:
            s_feats = s_feats[0]
            
        else:
            #concatenate layers into a tensor
            s_feats = torch.stack(s_feats)
            #now first dim is layer dimensions, take mean along there also to have (batch,1)
            s_feats = torch.mean(s_feats,dim=0) 

        #pass through ReLU to help
        s_feats = self.final_relu(s_feats)

        return s_feats

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


import timm
import torch.nn as nn
from icecream import ic

class CLIP_lpips_Unet(nn.Module):
    def __init__(self, clip_name:str, device:str):
        '''
        Lpips but with CLIP backbone
        '''
        super(CLIP_lpips_Unet, self).__init__()
        #load clip model
        self.clip = timm.create_model(clip_name, pretrained=True)
        self.clip.eval()
        self.clip.to(device)

        #Define clip hooks to get layer hidden features
        self.wanted_layers = ["stem.conv3"] + [f"stages.{s}.{2}.act" for s in range(4)] 
        print(self.wanted_layers)
        
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
        torch.save(self.decoder.state_dict(), path)

    def load_model(self,path:str):
        self.decoder.load_state_dict(torch.load(path, weights_only=True))
        