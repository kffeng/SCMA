import mindspore as ms
from mindspore import nn
import warnings
import math
import numpy as np
import mindspore.ops as ops
from mindspore.common.initializer import initializer, TruncatedNormal, Zero, One
from itertools import repeat
import collections.abc
from functools import partial
import mindspore.common.dtype as mstype
from models.helpers import load_pretrained
from models.vit_utils import DropPath,get_sinusoid_encoding_table
from models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from scipy.ndimage import zoom
from mindspore import ms_function
from mindspore import Tensor




class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0,
                 dtype=ms.float32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init="uniform").to_float(dtype)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features, weight_init="uniform").to_float(dtype)
        self.drop = nn.Dropout(1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, with_qkv=True,
                 compute_dtype=ms.float32):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dtype = compute_dtype
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.batchMatMul = ms.ops.BatchMatMul()
        self.cast = ms.ops.Cast()
        self.softmax = nn.Softmax(axis=-1).to_float(ms.float16)
        if self.with_qkv:
            self.qkv = nn.Dense(dim, dim * 3, weight_init="uniform", has_bias=qkv_bias).to_float(self.dtype)
            self.proj = nn.Dense(dim, dim, weight_init="uniform").to_float(self.dtype)
            self.proj_drop = nn.Dropout(1 - proj_drop)
        self.attn_drop = nn.Dropout(1 - attn_drop)

    def construct(self, x, return_attn=False):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose((2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).transpose((0, 2, 1, 3))
            q, k, v = qkv, qkv, qkv

        # q = q.astype(ms.float16)
        # k = k.astype(ms.float16)
        attn = self.batchMatMul(q, k.transpose(0, 1, 3, 2)) * self.scale
        # attn = attn.astype(ms.float16)
        attn = self.softmax(attn)
        # attn = attn.astype(ms.float32)
        attn = self.attn_drop(attn)
        attn = self.cast(attn, ms.float16)

        x = self.batchMatMul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time',
                 dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.attention_type = attention_type
        self.class_tokens = 0
        self.dim = dim
        self.mean = ops.ReduceMean(keep_dims=True)
        assert (attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])

        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, compute_dtype=self.dtype)
        # Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer((dim,)).to_float(ms.float32)
            self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, compute_dtype=self.dtype)
            self.temporal_fc = nn.Dense(dim, dim, weight_init="uniform").to_float(self.dtype)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer((dim,)).to_float(ms.float32)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       dtype=self.dtype)

    def construct(self, x, B, T, W, return_attn=False):
        num_spatial_tokens = (x.shape[1] - self.class_tokens) // T
        H = num_spatial_tokens // W
        
        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(F.cast(self.norm1(x), self.dtype)))
            x = x + self.drop_path(self.mlp(F.cast(self.norm2(x), self.dtype)))
            return x
        elif self.attention_type == 'divided_space_time':
            # Temporal
            if self.class_tokens == 0:
                xt = x[:, :, :]
            else:
                xt = x[:, 1:, :]

            xt = xt.reshape([B * H * W, T, -1])
            res_temporal = self.drop_path(self.temporal_attn(F.cast(self.temporal_norm1(xt), self.dtype)))
            res_temporal = res_temporal.reshape(B, H * W * T, -1)
            res_temporal = self.temporal_fc(res_temporal)
            if self.class_tokens == 0:
                xt = x[:, :, :] + res_temporal
            else:
                xt = x[:, 1:, :] + res_temporal
            # Spatial
            # init_cls_token = ms.ops.expand_dims(x[:, 0, :], 1)
            # cls_token = ms.numpy.tile(init_cls_token, (1, T, 1))
            # cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            # cls_token = cls_token.reshape(B * T, -1)
            # cls_token = ms.ops.expand_dims(cls_token, 1)

            aux_cls_token = None
            if self.class_tokens != 0:
                init_aux_cls_token = ms.ops.expand_dims(x[:, -1, :], 1)

                # aux_cls_token = init_aux_cls_token.repeat(1, T, 1)
                aux_cls_token = ms.numpy.tile(init_aux_cls_token, (1, T, 1))
                # aux_cls_token = rearrange(aux_cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
                aux_cls_token = aux_cls_token.reshape(B * T, -1)
                aux_cls_token = ms.ops.expand_dims(aux_cls_token, 1)
            else:
                aux_cls_token = None
            xs = xt
            # xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            xs=xs.reshape(B,H,W,T,-1).transpose(0,3,1,2,4)
            xs = xs.reshape(B * T, H * W, -1)

            # if aux_cls_token is None:
            #     xs = ms.ops.concat((cls_token, xs), axis=1)
            # else:
            #     xs = ms.ops.concat((cls_token, xs, aux_cls_token), axis=1)
            if return_attn:
                _, attn = self.attn(self.norm1(xs), return_attn=return_attn)
                return attn
            else:
                res_spatial = self.drop_path(self.attn(F.cast(self.norm1(xs), self.dtype)))

            # Taking care of CLS token
            # cls_token = res_spatial[:, 0, :]
            # cls_token = cls_token.reshape(B, T, -1)
            # cls_token = ms.Tensor(ms.numpy.mean(cls_token, 1, keepdims=True))
            # cls_token = self.mean(cls_token, 1) 
            if aux_cls_token is not None:
                aux_cls_token = res_spatial[:, -1, :]
                aux_cls_token = aux_cls_token.reshape(B, T, -1)

                # aux_cls_token = torch.mean(aux_cls_token, 1, True)  # averaging for every frame
                aux_cls_token = ms.Tensor(ms.numpy.mean(aux_cls_token, 1, keepdims=True))

            if aux_cls_token is None:
                res_spatial = res_spatial[:, :, :]
            else:
                res_spatial = res_spatial[:, 1:-1, :]
            # res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_spatial=res_spatial.reshape(B,T,H,W,-1).transpose(0,2,3,1,4)
            res_spatial = res_spatial.reshape(B, H * W * T, -1)

            res = res_spatial
            x = xt

            # Mlp
            # x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            # x = ms.ops.concat((init_cls_token, x), axis=1) + ms.ops.concat((cls_token, res), axis=1)
            x = x + res
            x = x + self.drop_path(self.mlp(F.cast(self.norm2(x), self.dtype)))
            return x
        return 1




class VisionTransformerDecoder(nn.Cell):
    """ Vision Transformer"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, decoder_embed_dim=384, depth=4,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=16,
                 attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(1 - dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.cast = ms.ops.Cast()
        self.dtype = ms.float16
        self.cat = ops.Concat(axis=1)
        self.expenddims = P.ExpandDims()
        self.squeeze_3 = P.Squeeze(3)


        # Positional Embeddings
        zeros = ms.ops.Zeros()
        num_patches=196
        # self.cls_token = ms.Parameter(zeros((1, 1, embed_dim), ms.float32))
        self.cls_token = self.cast(self.init_params((1, 1, decoder_embed_dim)),ms.float16)
    
        # self.pos_embed = ms.Parameter(zeros((1, num_patches + 1, embed_dim), ms.float16))

        self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim, dtype=self.dtype)
        self.pos_drop = nn.Dropout(keep_prob=1 - drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = get_sinusoid_encoding_table(num_frames, decoder_embed_dim, dtype=self.dtype)
            # self.time_embed = ms.Parameter(zeros((1, num_frames, embed_dim), ms.float16))
            self.time_drop = nn.Dropout(keep_prob=1 - drop_rate)

        # Attention Blocks
        dpr = [x for x in np.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        # dpr = [x for x in ops.LinSpace()(ms.Tensor(0.), ms.Tensor(drop_path_rate), self.depth).asnumpy()]
        self.blocks = nn.CellList([
            Block(
                dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type, dtype=self.dtype)
            for i in range(self.depth)])
        self.norm = norm_layer((decoder_embed_dim,)).to_float(ms.float32)

        # Classifier head
        self.encoder_to_decoder = nn.Dense(embed_dim, decoder_embed_dim, has_bias=False).to_float(ms.float16)

        self.mask_token = self.init_params((1, 1, decoder_embed_dim), initializer_type=Zero(), dtype=self.dtype)

        self.head_d = nn.Dense(decoder_embed_dim, embed_dim).to_float(ms.float32) if num_classes > 0 else nn.Identity()

        self.init_weights()

        if self.attention_type == 'divided_space_time':
            i = 0
            for _, cell in self.blocks.cells_and_names():
                cell_str = str(cell)
                if "Block" in cell_str:
                    if i > 0:
                        cell.temporal_fc.weight.set_data(
                            initializer(Zero(), cell.temporal_fc.weight.shape, cell.temporal_fc.weight.dtype))
                        cell.temporal_fc.bias.set_data(
                            initializer(Zero(), cell.temporal_fc.bias.shape, cell.temporal_fc.bias.dtype))
                    i += 1

    @staticmethod
    def init_params(
            shape,
            name=None,
            initializer_type=TruncatedNormal(sigma=0.02),
            dtype=ms.float32,
            requires_grad=True
    ):
        initial = initializer(initializer_type, shape, dtype)
        initial.init_data()
        return ms.Parameter(initial, name=name, requires_grad=requires_grad)

    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Zero(), cell.beta.shape, cell.beta.dtype))

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def forward_features(self, x_vis, ids_keep, ids_mask):

        B,N,C=x_vis.shape   # 2 197 768  197=4*49
        
        T=16
        W=H=6
        x_vis=x_vis.reshape(B,H,W,T,C).transpose(0,3,1,2,4)
        x_vis=x_vis.reshape(B*T,H*W,C)
        

        expand_pos_embed = self.pos_embed.broadcast_to((ids_keep.shape[0], -1, -1))
        
        pos_emd_vis = expand_pos_embed.gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1)).reshape(ids_keep.shape[0], -1, C)
        pos_emd_mask = expand_pos_embed.gather_elements(dim=1, index=ids_mask.expand_dims(-1).repeat(C, axis=-1)).reshape(ids_mask.shape[0], -1, C)


        x = ops.Concat(axis=1)((x_vis + pos_emd_vis, self.mask_token + pos_emd_mask))  # [B*T, N, C_d]  videomae是 b,T*N,c_d

        
        # Time Embeddings
        if self.attention_type != 'space_only':
            # cls_tokens = ms.ops.expand_dims(x[:B, 0, :], 1)
            # x = x[:, 1:]
            _,n,C=x.shape
            x=x.reshape(B,T,n,C).transpose(0,2,1,3)
            x = x.reshape(B*n, T, C)
            # Resizing time embeddings in case they don't match
            if T != self.time_embed.shape[1]:

                time_embed = self.time_embed.transpose(0, 2, 1)   #  1,768,8
                time_embed = self.expenddims(time_embed, 3)    #  1,768,8,1
                new_size = (T, 1)
                new_time_embed = P.ResizeNearestNeighbor(new_size)(time_embed)
                new_time_embed = self.squeeze_3(new_time_embed) 
                new_time_embed = new_time_embed.transpose(0, 2, 1)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = x.reshape(B, -1, C)

        # Attention blocks
        W=14
        for blk in self.blocks:
            x = blk(x, B, T, W)
            
        H=W=14
        x=x.reshape(B,H,W,T,C).transpose(0,3,1,2,4)
        x=x.reshape(B*T,H*W,C)
        x=x[:,-pos_emd_mask.shape[1]:,:]
        x=x.reshape(B,-1,C)
        
        x = self.head_d(self.norm(x))

        return x

        

    def construct(self, x_vis, ids_keep, ids_mask):  #x_vis [g1,g2]

        
        
        Bk,Tk,Lk=ids_keep.shape     #2,16,49
        Bm,Tm,Lm=ids_mask.shape     #2,16,147
        ids_keep=ids_keep.reshape(-1,Lk)
        ids_mask=ids_mask.reshape(-1,Lm)

        x_vis=self.encoder_to_decoder(x_vis)
        x = self.forward_features(x_vis, ids_keep=ids_keep, ids_mask=ids_mask)
 
        return x

