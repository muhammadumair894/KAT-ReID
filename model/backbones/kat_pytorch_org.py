"""
KATransformer (KAT) Backend for TransReID
Adapted from katransformer.py by Xingyi Yang
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from itertools import repeat
import copy
import sys
import os
import traceback
sys.setrecursionlimit(10000)

sys.path.append('/data_sata/ReID_Group/ReID_Group/KANTransfarmers/rational_kat_cu')
from kat_rational import KAT_Group
print("✓  KAT_Group loaded from rational_kat_cu (CUDA/Triton)")


from collections.abc import Iterable
import timm

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def kat_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    # Hardcode the KAT-specific parameters here
    activation = "swish"
    weight_init = "kan_mimetic"
    
    model = KATTransReID(
        img_size=img_size, 
        patch_size=16, 
        stride_size=stride_size, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        drop_path_rate=drop_path_rate, 
        drop_rate=drop_rate, 
        attn_drop_rate=attn_drop_rate, 
        camera=camera, 
        view=view, 
        sie_xishu=sie_xishu, 
        local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    return model



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# Adapter class that wraps KATransformer to provide the same interface as the ViT used in TransReID
# ... existing imports ...

class KATTransReID(nn.Module):
    def __init__(
            self, 
            img_size=(256, 128), 
            patch_size=16, 
            stride_size=16,
            embed_dim=768, 
            depth=12, 
            num_heads=12, 
            mlp_ratio=4., 
            qkv_bias=True,
            drop_path_rate=0.1, 
            drop_rate=0.0, 
            attn_drop_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sie_xishu=1.5,
            local_feature=False,
            camera=0,
            view=0,
            **kwargs
        ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size,
            in_chans=3, embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Initialize pos_embed and cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Initialize blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Create a KAT_Group activation function for the MLP
        # Use the CUDA-accelerated version with appropriate parameters
        def create_kat_activation(hidden_features):
            return KAT_Group(
                num_groups=8,  # Fixed number of groups
                order=(5, 4),  # polynomial order
                bias=True,     # use bias
                cuda=True      # enable CUDA acceleration
            )
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # SIE embedding for ReID
        if camera > 0 and view > 0:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is: {}, view number is: {}'.format(camera, view))
        elif camera > 0:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is: {}'.format(camera))
        elif view > 0:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('view number is: {}'.format(view))
        else:
            self.sie_embed = None
            
        self.sie_xishu = sie_xishu
        self.camera = camera
        self.view = view
        
        # Important: For compatibility with make_model.py
        #self.base = self
        object.__setattr__(self, "base", self)   # bypasses Module.__setattr__
        
        # Initialize weights
        self._init_weights_safe()
    
    def _init_weights_safe(self):
        """Safely initialize weights without excessive recursion"""
        # Initialize pos_embed and cls_token
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        # Manually initialize specific layers instead of using recursive apply
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            
            # Skip weight initialization for KAT_Group layers
            # They will handle their own initialization
            if isinstance(m, KAT_Group):
                continue

    def forward(self, x, camera_id=None, view_id=None, cam_label=None, view_label=None):
        """Forward pass with support for camera and view IDs.
        
        Args:
            x: Input tensor
            camera_id: Camera ID (deprecated, use cam_label instead)
            view_id: View ID (deprecated, use view_label instead)
            cam_label: Camera label (new parameter name)
            view_label: View label (new parameter name)
        """
        # For backward compatibility, use cam_label if provided, otherwise use camera_id
        if cam_label is not None:
            camera_id = cam_label
        if view_label is not None:
            view_id = view_label
            
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.sie_embed is not None:
            if camera_id is not None and view_id is not None:
                x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view + view_id]
            elif camera_id is not None:
                x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
            elif view_id is not None:
                x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
            else:
                x = x + self.pos_embed
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # Return features based on local_feature flag
        if self.local_feature:
            return x
        else:
            return x[:, 0]  # Return only cls token features

    def load_param(self, model_path):
        """Load pretrained model parameters from file."""
        param_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different pretrained model formats
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        # Process position embedding if needed
        if 'pos_embed' in param_dict:
            pos_embed_checkpoint = param_dict['pos_embed']
            embed_dim = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = 1  # cls token
            
            # Calculate original grid size
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            
            # Get new size from patch_embed
            new_size = (self.patch_embed.H, self.patch_embed.W)
            
            # Only interpolate if sizes don't match
            if orig_size != new_size[0] or orig_size != new_size[1]:
                print(f"Interpolating position embedding from {orig_size}x{orig_size} to {new_size[0]}x{new_size[1]}")
                
                # Extract tokens and interpolate
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:].reshape(
                    1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=new_size, mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                
                # Concatenate with class token
                new_pos_embed = torch.cat((pos_embed_checkpoint[:, :num_extra_tokens], pos_tokens), dim=1)
                param_dict['pos_embed'] = new_pos_embed
        
        # Manually load parameters to avoid recursion
        try:
            # Load cls_token and pos_embed directly
            if 'cls_token' in param_dict:
                self.cls_token.data.copy_(param_dict['cls_token'])
            if 'pos_embed' in param_dict:
                self.pos_embed.data.copy_(param_dict['pos_embed'])
            
            # Load patch_embed parameters
            if 'patch_embed.proj.weight' in param_dict:
                self.patch_embed.proj.weight.data.copy_(param_dict['patch_embed.proj.weight'])
            if 'patch_embed.proj.bias' in param_dict:
                self.patch_embed.proj.bias.data.copy_(param_dict['patch_embed.proj.bias'])
            
            # Load transformer blocks parameters
            for i, block in enumerate(self.blocks):
                block_prefix = f'blocks.{i}.'
                # Load norm1 parameters
                if f'{block_prefix}norm1.weight' in param_dict:
                    block.norm1.weight.data.copy_(param_dict[f'{block_prefix}norm1.weight'])
                if f'{block_prefix}norm1.bias' in param_dict:
                    block.norm1.bias.data.copy_(param_dict[f'{block_prefix}norm1.bias'])
                
                # Load attention parameters
                if f'{block_prefix}attn.qkv.weight' in param_dict:
                    block.attn.qkv.weight.data.copy_(param_dict[f'{block_prefix}attn.qkv.weight'])
                if f'{block_prefix}attn.qkv.bias' in param_dict:
                    block.attn.qkv.bias.data.copy_(param_dict[f'{block_prefix}attn.qkv.bias'])
                if f'{block_prefix}attn.proj.weight' in param_dict:
                    block.attn.proj.weight.data.copy_(param_dict[f'{block_prefix}attn.proj.weight'])
                if f'{block_prefix}attn.proj.bias' in param_dict:
                    block.attn.proj.bias.data.copy_(param_dict[f'{block_prefix}attn.proj.bias'])
                
                # Load norm2 parameters
                if f'{block_prefix}norm2.weight' in param_dict:
                    block.norm2.weight.data.copy_(param_dict[f'{block_prefix}norm2.weight'])
                if f'{block_prefix}norm2.bias' in param_dict:
                    block.norm2.bias.data.copy_(param_dict[f'{block_prefix}norm2.bias'])
                
                # Load MLP parameters - handle KAT-specific structure
                if f'{block_prefix}mlp.fc1.weight' in param_dict:
                    block.mlp.fc1.weight.data.copy_(param_dict[f'{block_prefix}mlp.fc1.weight'])
                if f'{block_prefix}mlp.fc1.bias' in param_dict:
                    block.mlp.fc1.bias.data.copy_(param_dict[f'{block_prefix}mlp.fc1.bias'])
                if f'{block_prefix}mlp.fc2.weight' in param_dict:
                    block.mlp.fc2.weight.data.copy_(param_dict[f'{block_prefix}mlp.fc2.weight'])
                if f'{block_prefix}mlp.fc2.bias' in param_dict:
                    block.mlp.fc2.bias.data.copy_(param_dict[f'{block_prefix}mlp.fc2.bias'])
                

            
            # Load final norm layer
            if 'norm.weight' in param_dict:
                self.norm.weight.data.copy_(param_dict['norm.weight'])
            if 'norm.bias' in param_dict:
                self.norm.bias.data.copy_(param_dict['norm.bias'])
            
            print(f"Successfully loaded pretrained model from {model_path}")
        except Exception as e:
            print(f"Error during manual parameter loading: {e}")
            traceback.print_exc()
        
        return self
    

# Patch Embedding with stride support for TransReID
class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding with support for different strides
    Adapted from timm's implementation
    """
    def __init__(self, img_size=(256, 128), patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        
        self.H, self.W = img_size[0] // stride_size[0], img_size[1] // stride_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# Create factory functions similar to the ViT ones
def kat_tiny_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = KATTransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def kat_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = KATTransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def kat_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    # Hardcode the KAT-specific parameters here
    activation = "swish"
    weight_init = "kan_mimetic"
    
    model = KATTransReID(
        img_size=img_size, 
        patch_size=16, 
        stride_size=stride_size, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        drop_path_rate=drop_path_rate, 
        drop_rate=drop_rate, 
        attn_drop_rate=attn_drop_rate, 
        camera=camera, 
        view=view, 
        sie_xishu=sie_xishu, 
        local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    
    # Monkey patch the to() method to use our custom implementation
    original_to = model.to
    model.to = lambda device: model.custom_to(device)
    
    return model

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KAN(nn.Module):                       # Kolmogorov–Arnold block
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super().__init__()
        hid = int(dim * mlp_ratio)
        assert hid % 8 == 0, "hidden dim must be divisible by 8 for group rational"
        self.act1 = KAT_Group(mode='identity')       # keeps ViT weights valid
        self.fc1  = nn.Linear(dim, hid)
        self.norm = nn.LayerNorm(hid)
        self.act2 = KAT_Group(mode='swish')          # Swish-like rational
        self.fc2  = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # Use drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = KAN(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x