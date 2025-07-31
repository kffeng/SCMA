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
from models.vit_utils import DropPath
from models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from scipy.ndimage import zoom
from mindspore import ms_function


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )
}


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
        self.class_tokens = 1
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
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path = nn.Identity()
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
            if self.class_tokens == 1:
                xt = x[:, 1:, :]
            else:
                xt = x[:, 1:-1, :]

            xt = xt.reshape([B * H * W, T, -1])
            res_temporal = self.drop_path(self.temporal_attn(F.cast(self.temporal_norm1(xt), self.dtype)))
            res_temporal = res_temporal.reshape(B, H * W * T, -1)
            res_temporal = self.temporal_fc(res_temporal)
            if self.class_tokens == 1:
                xt = x[:, 1:, :] + res_temporal
            else:
                xt = x[:, 1:-1, :] + res_temporal
            # Spatial
            init_cls_token = ms.ops.expand_dims(x[:, 0, :], 1)
            cls_token = ms.numpy.tile(init_cls_token, (1, T, 1))
            cls_token = cls_token.reshape(B * T, -1)
            cls_token = ms.ops.expand_dims(cls_token, 1)

            xs = xt
            xs=xs.reshape(B,H,W,T,-1).transpose(0,3,1,2,4)
            xs = xs.reshape(B * T, H * W, -1)
            xs = ms.ops.concat((cls_token, xs), axis=1)

            x, attn = self.attn(self.norm1(xs), return_attn=return_attn)
            res_spatial = self.drop_path(x)

            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = cls_token.reshape(B, T, -1)
            # cls_token = ms.Tensor(ms.numpy.mean(cls_token, 1, keepdims=True))
            cls_token = self.mean(cls_token, 1)

            res_spatial = res_spatial[:, 1:, :]
            res_spatial=res_spatial.reshape(B,T,H,W,-1).transpose(0,2,3,1,4)
            res_spatial = res_spatial.reshape(B, H * W * T, -1)

            res = res_spatial
            x = xt

            # Mlp
            x = ms.ops.concat((init_cls_token, x), axis=1) + ms.ops.concat((cls_token, res), axis=1)
            x = x + self.drop_path(self.mlp(F.cast(self.norm2(x), self.dtype)))
            return x,attn
        return 1


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True,
                              pad_mode="valid")

    def construct(self, x):
        B, C, T, H, W = x.shape
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape([-1, C, H, W])
        x = self.proj(x)
        W = x.shape[-1]
        x = ops.reshape(x, (x.shape[0], x.shape[1], -1))
        x = ops.transpose(x, (0, 2, 1))
        return x, T, W


class VisionTransformer(nn.Cell):
    """ Vision Transformer"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=16,
                 attention_type='divided_space_time', dropout=0. ,masked_im_modeling=False):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(1 - dropout)
        self.num_classes = num_classes
        self.masked_im_modeling=masked_im_modeling
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cast = ms.ops.Cast()
        self.dtype = ms.float16
        self.cat = ops.Concat(axis=1)
        self.expenddims = P.ExpandDims()
        self.squeeze_3 = P.Squeeze(3)
        # self.fast_nearst_T = P.ResizeNearestNeighbor((64, 1))
        # self.fast_nearst_HW = P.ResizeNearestNeighbor((6, 6))

        # Positional Embeddings
        zeros = ms.ops.Zeros()
        # self.cls_token = ms.Parameter(zeros((1, 1, embed_dim), ms.float32))
        self.cls_token = self.init_params((1, 1, embed_dim))
        # self.pos_embed = ms.Parameter(zeros((1, num_patches + 1, embed_dim), ms.float32))
        self.pos_embed = self.init_params((1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(keep_prob=1 - drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = ms.Parameter(zeros((1, num_frames, embed_dim), ms.float32))
            self.time_drop = nn.Dropout(keep_prob=1 - drop_rate)

        # Attention Blocks
        dpr = [x for x in np.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        # dpr = [x for x in ops.LinSpace()(ms.Tensor(0.), ms.Tensor(drop_path_rate), self.depth).asnumpy()]
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type, dtype=self.dtype)
            for i in range(self.depth)])
        self.norm = norm_layer((embed_dim,)).to_float(ms.float32)

        # Classifier head
        self.head = nn.Dense(embed_dim, num_classes, weight_init="uniform").to_float(self.dtype)

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

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, ids_keep_g=None,ids_mask_g=None,ids_keep_l=None):
        B = x.shape[0]
        
        x, T, W = self.patch_embed(x)
        H=W
        cls_tokens = self.cls_token.broadcast_to((x.shape[0], -1, -1))
        x = self.cat((cls_tokens, x))

        # resizing the positional embeddings in case they don't match the input at inference
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self.pos_embed
            # cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            cls_pos_embed = ms.ops.expand_dims(pos_embed[0, 0, :], 0)
            cls_pos_embed = ms.ops.expand_dims(cls_pos_embed, 1)
            # other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            other_pos_embed = ms.ops.expand_dims(pos_embed[0, 1:, :], 0)
            other_pos_embed = ms.ops.transpose(other_pos_embed, (0, 2, 1))

            V = int(other_pos_embed.shape[2] ** 0.5)
            H = x.shape[1] // W
            other_pos_embed = other_pos_embed.reshape(1, x.shape[2], V, V)
            # other_pos_embed = other_pos_embed.reshape(1, 768, T, T)

            # new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            # new_pos_embed = ms.ops.interpolate(other_pos_embed, sizes=(H, W), mode="bilinear")
            new_size = (H, W)
            new_pos_embed = P.ResizeNearestNeighbor(new_size)(other_pos_embed)
            new_pos_embed = new_pos_embed.reshape(new_pos_embed.shape[0], new_pos_embed.shape[1], -1)
            new_pos_embed = ms.ops.transpose(new_pos_embed, (0, 2, 1))
            new_pos_embed = ms.ops.concat((cls_pos_embed, new_pos_embed), axis=1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        target=x


        #mask
        if self.masked_im_modeling :
        
            cls_t = ms.ops.expand_dims(x[:, 0, :], 1)
            
            x = x[:, 1:, :]   # 128,196,768
            BT,_,C=x.shape
            
            Bk,Tk,Lk=ids_keep_g.shape     #2,16,49
            Bm,Tm,Lm=ids_mask_g.shape     #2,16,147
            
            ids_keep_g=ids_keep_g.reshape(-1,Lk)
            ids_mask_g=ids_mask_g.reshape(-1,Lm)
            
            ids_keep=[]
            if B==4:   #  2是bs
                W=6
                ids_keep=ids_keep_g
            elif B==4*8:   #  8是bs   4是每个video为4个clip
                W=7
                ids_keep=ids_keep_l
            else:
                print("Student Global Mask Error ! ! !")
            x_vis = x.gather_elements(dim=1, index=ids_keep.expand_dims(-1).repeat(C, axis=-1))

            x = self.cat((cls_t, x_vis))


        # Time Embeddings
        if self.attention_type != 'space_only':
            # cls_tokens = x[:B, 0, :].unsqueeze(1)
            cls_tokens = ms.ops.expand_dims(x[:B, 0, :], 1)
            x = x[:, 1:]
            _,n,C=x.shape
            # x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
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
            # x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = x.reshape(B, -1, C)
            x = ms.ops.concat((cls_tokens, x), axis=1)
            # x = torch.cat((cls_tokens, x), dim=1)

        # Attention blocks
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, attn= blk(x, B, T, W, return_attn=True)
            else:
                # return attention of the last block
                x, attn = blk(x, B, T, W,return_attn=True)
                # x = self.norm(x)    # B L C   --> B H W T C
                
                if self.masked_im_modeling:
                    #student
                    x = self.norm(x)
                    return x
                else:
                    _,_,C=x.shape
                    # 老师的输出不norm
                    # cls_token=x[:,0]
                    # cls_token = self.norm(cls_token)
                    # 老师的输出norm
                    x=self.norm(x)
                    cls_token=x[:,0]
                    #重建特征
                    x=x[:,1:,:]
                    x=x.reshape(B,H,W,T,C).transpose(0,3,1,2,4)
                    x_full=x.reshape(B*T,H*W,C)
                    #重建像素
                    # mean = target.mean(axis=-1, keep_dims=True)
                    # var = target.var(axis=-1, keepdims=True)
                    # target = (target - mean) / (var ** 0.5 + 1.0e-6)
                    return cls_token, x_full
        return 1



    def construct(self, x ,ids_keep_g=None,ids_mask_g=None,ids_keep_l=None):
        if ids_keep_g==None:
            cls_token, x_full= self.forward_features(x)
            return cls_token, x_full
        elif ids_keep_g!=None:
            x=self.forward_features(x,ids_keep_g=ids_keep_g,ids_mask_g=ids_keep_g,ids_keep_l=ids_keep_l)
            return x
        else:
            print("Output Error ! ! !")
        return 1


    def get_intermediate_layers(self, x, n=1):
        x = self.forward_features(x, get_all=True)
        return [x, ]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            v = ms.Parameter(v, requires_grad=True)
        out_dict[k] = v
    return out_dict


def get_vit_base_patch16_224(cfg, no_head=False, masked_im_modeling=False,**kwargs):
    patch_size = 16
    vit = VisionTransformer(img_size=cfg.TRAIN_CROP_SIZE, num_classes=cfg.NUM_CLASSES,
                            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                            qkv_bias=True, norm_layer=nn.LayerNorm, drop_rate=0.0,
                            attn_drop_rate=0.0, drop_path_rate=0.1, num_frames=16,
                            attention_type=cfg.ATTENTION_TYPE,masked_im_modeling=masked_im_modeling,**kwargs)
    vit.attention_type = cfg.ATTENTION_TYPE
    vit.default_cfg = default_cfgs['vit_base_patch16_224']
    vit.num_patches = (cfg.TRAIN_CROP_SIZE // patch_size) * (cfg.TRAIN_CROP_SIZE // patch_size)
    pretrained_model = cfg.PRETRAINED_MODEL
    if pretrained_model:
        load_pretrained(vit, args=cfg, num_classes=vit.num_classes, in_chans=kwargs.get('in_chans', 3),
                        filter_fn=_conv_filter, img_size=cfg.TRAIN_CROP_SIZE, num_patches=vit.num_patches,
                        attention_type=vit.attention_type, pretrained_model=pretrained_model)
    if no_head:
        vit.head = None
    return vit


def parse_yaml(parser, yaml_path):
    import yaml
    with open(yaml_path, 'r') as fin:
        cfg = yaml.load(fin.read(), Loader=yaml.FullLoader)

    for k, v in cfg.items():
        parser.add_argument('--' + k, default=v)
    return parser


if __name__ == '__main__':
    from utils.parser import parse_args

    opt = parse_args()
    opt.cfg_file = "./configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = parse_yaml(opt, opt.cfg_file)
    config.TRAIN_CROP_SIZE = 224
    config.NUM_CLASSES = 400
    config.NUM_FRAMES = 8
    config.PRETRAINED_MODEL = ""
    config.ATTENTION_TYPE = 'divided_space_time'
    model = get_vit_base_patch16_224(cfg=config)

    sample = ms.ops.Ones()((2, 3, 8, 224, 224), ms.float32)
    out1 = model(sample)
    print(out1.shape)
