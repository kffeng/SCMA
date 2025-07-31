# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

# import torch
# import torch.nn as nn

from utils.utils import trunc_normal_
import mindspore
import mindspore.ops as P
import mindspore.nn as nn
import numpy as np
from utils.weight import WeightNorm
from utils.dense import dense_
from mindspore.common.initializer import initializer,TruncatedNormal,Zero,One
from mindspore import ms_function
#已改
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    uniformreal = P.UniformReal(seed=2)
    # random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = keep_prob +uniformreal(shape, dtype=x.dtype, device=x.device)
    #不知道怎么改
    random_tensor.floor_()  # binarize
    # output = x.div(keep_prob) * random_tensor
    output = P.div(x,keep_prob) * random_tensor
    return output

#已改
class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

#已改
class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init="uniform")
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features, weight_init="uniform")
        self.drop = nn.Dropout(1 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn

#已改
class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3,weight_init='uniform', has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.softmax=nn.Softmax(axis=-1)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv=self.reshape(qkv,(B, N, 3, self.num_heads, C // self.num_heads))
        qkv=self.transpose(qkv,(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        k=self.transpose(k,(-2, -1))
        attn = (q @ k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # @好像需要改
        x = (attn @ v)  
        x=self.transpose(x,(1, 2))
        x=self.reshape(x,(B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn



# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, return_attention=False):
#         y, attn = self.attn(self.norm1(x))
#         if return_attention:
#             return attn
#         x = x + self.drop_path(y)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

#已改
class Block(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #nn.Identity()未改
        if drop_path > 0. :
            self.drop_path = DropPath(drop_path) 
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

#已改
class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,has_bias=True)

    def construct(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = self.reshape(x, (B, C, H * W))
        x = self.transpose(x, (0, 2, 1))
        return x


# class VisionTransformer(nn.Module):
#     """ Vision Transformer """
#     def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim

#         self.patch_embed = PatchEmbed(
#             img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         # Classifier head
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def interpolate_pos_encoding(self, x, w, h):
#         npatch = x.shape[1] - 1
#         N = self.pos_embed.shape[1] - 1
#         if npatch == N and w == h:
#             return self.pos_embed
#         class_pos_embed = self.pos_embed[:, 0]
#         patch_pos_embed = self.pos_embed[:, 1:]
#         dim = x.shape[-1]
#         w0 = w // self.patch_embed.patch_size
#         h0 = h // self.patch_embed.patch_size
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#             mode='bicubic',
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#     def prepare_tokens(self, x):
#         B, nc, w, h = x.shape
#         x = self.patch_embed(x)  # patch linear embedding

#         # add the [CLS] token to the embed patch tokens
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # add positional encoding to each token
#         x = x + self.interpolate_pos_encoding(x, w, h)

#         return self.pos_drop(x)

#     def forward(self, x):
#         x = self.prepare_tokens(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         return x[:, 0]

#     def get_intermediate_layers(self, x, n=1):
#         x = self.prepare_tokens(x)
#         # we return the output tokens from the `n` last blocks
#         output = []
#         for i, blk in enumerate(self.blocks):
#             x = blk(x)
#             if len(self.blocks) - i <= n:
#                 output.append(self.norm(x))
#         return output

#已改
class VisionTransformer(nn.Cell):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mindspore.Parameter(P.Zeros()((1,1,embed_dim),mindspore.float32))
        self.pos_embed = mindspore.Parameter(P.Zeros()((1, num_patches + 1, embed_dim),mindspore.float32))
         #keep_prob参数为保留参数的概率。
        self.pos_drop = nn.Dropout(keep_prob=(1.0-drop_rate))

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #不确定要不要转化为tensor
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]   # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        # self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if num_classes > 0:
            self.head = nn.Dense(embed_dim, num_classes)


        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                # nn.init.constant_(m.bias, 0)
                mindspore.common.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            mindspore.common.initializer.Constant(0)(m.bias)
            mindspore.common.initializer.Constant(1.0)(m.weight)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        # TODO nn.functional.interpolate   mindspore没有
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        #mindspore  2.0版本有了permute
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return P.Concat(axis=1)(P.ExpandDims()(class_pos_embed,0), patch_pos_embed)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens=P.function.broadcast_to(self.cls_token,(B, -1, -1))
        # x = torch.cat((cls_tokens, x), dim=1)
        x=P.Concat(axis=1)(cls_tokens, x)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def construct(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


#已改
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

#已改
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

#已改
def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# class DINOHead(nn.Module):
#     def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
#         super().__init__()
#         nlayers = max(nlayers, 1)
#         if nlayers == 1:
#             self.mlp = nn.Linear(in_dim, bottleneck_dim)
#         else:
#             layers = [nn.Linear(in_dim, hidden_dim)]
#             if use_bn:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.GELU())
#             for _ in range(nlayers - 2):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#                 if use_bn:
#                     layers.append(nn.BatchNorm1d(hidden_dim))
#                 layers.append(nn.GELU())
#             layers.append(nn.Linear(hidden_dim, bottleneck_dim))
#             self.mlp = nn.Sequential(*layers)
#         self.apply(self._init_weights)
#         self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
#         self.last_layer.weight_g.data.fill_(1)
#         if norm_last_layer:
#             self.last_layer.weight_g.requires_grad = False

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.mlp(x)
#         x = nn.functional.normalize(x, dim=-1, p=2)
#         x = self.last_layer(x)
#         return x


#已改
# class DINOHead1(nn.Cell):
#     def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
#         super().__init__()
#         nlayers = max(nlayers, 1)
#         self.cast = mindspore.ops.Cast()
#         if nlayers == 1:
#             self.mlp = nn.Dense(in_dim, bottleneck_dim).to_float(mindspore.float16)
#         else:
#             layers = [nn.Dense(in_dim, hidden_dim).to_float(mindspore.float16)]
#             if use_bn:
#                 layers.append(nn.BatchNorm1d(hidden_dim).to_float(mindspore.float16))
#             layers.append(nn.GELU().to_float(mindspore.float16))
#             for _ in range(nlayers - 2):
#                 layers.append(nn.Dense(hidden_dim, hidden_dim).to_float(mindspore.float16))
#                 if use_bn:
#                     layers.append(nn.BatchNorm1d(hidden_dim).to_float(mindspore.float16))
#                 layers.append(nn.GELU().to_float(mindspore.float16))
#             layers.append(nn.Dense(hidden_dim, bottleneck_dim).to_float(mindspore.float16))
#             self.mlp = nn.SequentialCell(*layers).to_float(mindspore.float16)
#         # self.apply(self._init_weights)
#         for m in self.cells_and_names():
#             self._init_weights(m[1])
#         # TODO 没有
#         self.last_layer =WeightNorm(nn.Dense(bottleneck_dim, out_dim, has_bias=False).to_float(mindspore.float16))
#         # self.last_layer.weight_g.data.fill_(1)
#         # if norm_last_layer:
#         #     self.last_layer.weight_g.requires_grad = False

#     def _init_weights(self, m):
#         if isinstance(m, nn.Dense):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Dense) and m.bias is not None:
#                 # nn.init.constant_(m.bias, 0)
#                 mindspore.common.initializer.Constant(0)(m.bias)

#     def construct(self, x):
#         x = self.cast(x, mindspore.float16)
#         x = self.mlp(x)
#         # x = nn.functional.normalize(x, dim=-1, p=2)
#         l2_normalize = P.L2Normalize(axis=-1,epsilon=1e-12)
#         x = self.cast(x, mindspore.float16)
#         x=l2_normalize(x)
#         x = self.cast(x, mindspore.float16)
#         x = self.last_layer(x)
#         x = self.cast(x, mindspore.float32)
#         return x


class DINOHead(nn.Cell):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Dense(in_dim, bottleneck_dim)
        else:
            layers = [nn.Dense(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU(approximate=False))
            for _ in range(nlayers - 2):
                layers.append(nn.Dense(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU(approximate=False))
            layers.append(nn.Dense(hidden_dim, bottleneck_dim))
            self.mlp = nn.SequentialCell(layers)
        self.init_weights()

        self.last_layer =dense_(bottleneck_dim, out_dim)
        self.norm = P.L2Normalize(-1,epsilon=1e-12)

    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(0.0, cell.bias.shape))

    def construct(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        x = self.last_layer(x)
        return x







#已改
class MultiDINOHead(nn.Cell):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Dense(in_dim, bottleneck_dim)
            self.aux_mlp = nn.Dense(in_dim, bottleneck_dim)
        else:
            layers = [nn.Dense(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Dense(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Dense(hidden_dim, bottleneck_dim))
            self.mlp = nn.SequentialCell(*layers)

            aux_layers = [nn.Dense(in_dim, hidden_dim)]
            if use_bn:
                aux_layers.append(nn.BatchNorm1d(hidden_dim))
            aux_layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                aux_layers.append(nn.Dense(hidden_dim, hidden_dim))
                if use_bn:
                    aux_layers.append(nn.BatchNorm1d(hidden_dim))
                aux_layers.append(nn.GELU())
            aux_layers.append(nn.Dense(hidden_dim, bottleneck_dim))
            self.aux_mlp = nn.SequentialCell(*aux_layers)

        # self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Dense(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        # TODO 没有
        self.aux_last_layer = nn.utils.weight_norm(nn.LinDenseear(bottleneck_dim, out_dim, bias=False))
        self.aux_last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.aux_last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                # nn.init.constant_(m.bias, 0)
                mindspore.common.initializer.Constant(0)(m.bias)
    # @ms_function
    def construct(self, x):
        rgb_x, aux_x = x[0], x[1]

        rgb_x = self.mlp(rgb_x)
        # rgb_x = nn.functional.normalize(rgb_x, dim=-1, p=2)
        rgb_x=P.L2Normalize(axis=-1,epsilon=1e-12)(rgb_x)
        rgb_x = self.last_layer(rgb_x)

        aux_x = self.aux_mlp(aux_x)
        aux_x = P.L2Normalize(axis=-1,epsilon=1e-12)(aux_x)
        aux_x = self.aux_last_layer(aux_x)
        return rgb_x, aux_x