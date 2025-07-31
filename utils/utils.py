# # Copyright (c) Facebook, Inc. and its affiliates.
# # 
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# # 
# #     http://www.apache.org/licenses/LICENSE-2.0
# # 
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Misc functions.

# Mostly copy-paste from torchvision references or other public repos like DETR:
# https://github.com/facebookresearch/detr/blob/master/util/misc.py
# """
# import mindspore
# import os
# import sys
# from mindspore import Tensor
# import mindspore.ops as P
# import mindspore.nn as nn
# from mindspore import context
# from mindspore.common import set_seed
# from mindspore.communication.management import init, get_group_size, get_rank
# import argparse
# import time
# import math
# import random
# import datetime
# import subprocess
# from collections import defaultdict, deque

# import numpy as np
# from PIL import ImageFilter, ImageOps
# import warnings
# from mindspore.nn import Cell



# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule

# def sgd_lr(base_value,final_value,epochs,niter_per_ep,warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     lr1=np.linspace(base_value, base_value, niter_per_ep*10)
#     schedule = np.concatenate((warmup_schedule,lr1))
#     lr2=np.linspace(base_value/10, base_value/10, niter_per_ep*3)
#     schedule = np.concatenate((schedule, lr2))
#     lr3=np.linspace(base_value/100, base_value/100, niter_per_ep*2)
#     schedule = np.concatenate((schedule, lr3))
#     lr4=np.linspace(base_value/1000, base_value/1000, niter_per_ep*(epochs-15))
#     schedule = np.concatenate((schedule, lr4))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule




# class cosine_scheduler_weightDecay(Cell):
#     def __init__(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
#         super(cosine_scheduler_weightDecay, self).__init__()
#         self.warmup_schedule = np.array([])
#         self.warmup_iters = warmup_epochs * niter_per_ep
#         if warmup_epochs > 0:
#             self.warmup_schedule = np.linspace(start_warmup_value, base_value, self.warmup_iters)

#         iters = np.arange(epochs * niter_per_ep - self.warmup_iters)
#         self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

#         self.schedule = np.concatenate((self.warmup_schedule, self.schedule))
#         assert len(self.schedule) == epochs * niter_per_ep
#         self.schedule = Tensor(self.schedule,mindspore.float32)

#     def construct(self, global_step):
        
#         return self.schedule[global_step]




















# # ok
# class GaussianBlur(object):
#     """
#     Apply Gaussian Blur to the PIL image.
#     """
#     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max

#     def __call__(self, img):
#         do_it = random.random() <= self.prob
#         if not do_it:
#             return img

#         return img.filter(
#             ImageFilter.GaussianBlur(
#                 radius=random.uniform(self.radius_min, self.radius_max)
#             )
#         )


# class Solarization(object):
#     """
#     Apply Solarization to the PIL image.
#     """
#     def __init__(self, p):
#         self.p = p

#     def __call__(self, img):
#         if random.random() < self.p:
#             return ImageOps.solarize(img)
#         else:
#             return img

# # TODO mindspore的梯度不存在tensor中
# def clip_gradients(model, clip):
#     norms = []
#     for name, p in model.parameters_and_names():
#         p.grad = None
#         if p.grad is not None:
#             param_norm = p.grad.data.norm(2)
#             norms.append(param_norm.item())
#             clip_coef = clip / (param_norm + 1e-6)
#             if clip_coef < 1:
#                 p.grad.data.mul_(clip_coef)
#     return norms

# # 已改
# def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
#     if epoch >= freeze_last_layer:
#         return
#     for n, p in model.parameters_and_names():
#         if "last_layer" in n:
#             p.grad = None

# #已改
# def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
#     """
#     Re-start from checkpoint
#     """
#     if not os.path.isfile(ckp_path):
#         return
#     print("Found checkpoint at {}".format(ckp_path))

#     # open checkpoint file
#     # checkpoint = torch.load(ckp_path, map_location="cpu")
#     checkpoint=mindspore.load_checkpoint(ckp_path)

#     # key is what to look for in the checkpoint file
#     # value is the object to load
#     # example: {'state_dict': model}
#     for key, value in kwargs.items():
#         if key in checkpoint and value is not None:
#             try:
#                 # msg = value.load_state_dict(checkpoint[key], strict=False)
#                 #TODO 不确定
#                 msg=mindspore.load_param_into_net(checkpoint[key],value)
#                 print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
#             except TypeError:
#                 try:
#                     # msg = value.load_state_dict(checkpoint[key])
#                     msg=mindspore.load_param_into_net(checkpoint[key],value)
#                     print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
#                 except ValueError:
#                     print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
#         else:
#             print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

#     # re load variable important for the run
#     if run_variables is not None:
#         for var_name in run_variables:
#             if var_name in checkpoint:
#                 run_variables[var_name] = checkpoint[var_name]





# # ok
# def bool_flag(s):
#     """
#     Parse boolean arguments from the command line.
#     """
#     FALSY_STRINGS = {"off", "false", "0"}
#     TRUTHY_STRINGS = {"on", "true", "1"}
#     if s.lower() in FALSY_STRINGS:
#         return False
#     elif s.lower() in TRUTHY_STRINGS:
#         return True
#     else:
#         raise argparse.ArgumentTypeError("invalid value for a boolean flag")


# # ok
# def fix_random_seeds(seed=31):
#     """
#     Fix random seeds.
#     """
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed)
#     # np.random.seed(seed)
#     mindspore.set_seed(seed)



# # TODO
# def is_main_process():
#     # return get_rank() == 0
#     return True

# # TODO
# def save_on_master(*args, **kwargs):
#     if is_main_process():
#         mindspore.save_checkpoint(*args, **kwargs)


# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print

# MODE = {"PYNATIVE_MODE": context.PYNATIVE_MODE,
#         "GRAPH_MODE": context.GRAPH_MODE}
# def cloud_context_init(
#                         seed=0,
#                        use_parallel=True,
#                        context_config=None,
#                        parallel_mode=None):
#     np.random.seed(seed)
#     set_seed(seed)
#     # mode_config = context.GRAPH_MODE
#     context_config["mode"] = MODE[context_config["mode"]]
#     rank_id, device_num = 0, 1
#     if use_parallel:
#         device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
#         context_config["device_id"] = device_id
#         parallel_mode=context.ParallelMode.DATA_PARALLEL
#         context.set_context(**context_config)
#         init()
#         rank_id = get_rank()  # local_rank
#         device_num = get_group_size()  # world_size
#         context.set_auto_parallel_context(
#             device_num=device_num, gradients_mean=True,parallel_mode=parallel_mode)
#     os.environ['MOX_SILENT_MODE'] = '1'
#     return rank_id, device_num



# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # type: (Tensor, float, float, float, float) -> Tensor
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)



# class Identity(nn.Cell):

#     def construct(self, x):
#         return x


# class MultiCropWrapper(nn.Cell):
#     """
#     Perform forward pass separately on each resolution input.
#     The inputs corresponding to a single resolution are clubbed and single
#     forward is run on the same resolution inputs. Hence we do several
#     forward passes = number of different resolutions used. We then
#     concatenate all the output features and run the head forward on these
#     concatenated features.
#     """
#     def __init__(self, backbone, vary_fr=False):
#         super(MultiCropWrapper, self).__init__()
#         # disable layers dedicated to ImageNet labels classification
#         if hasattr(backbone, 'fc'):
#             backbone.fc, backbone.head = Identity(), Identity()
#         self.backbone = backbone
#         self.vary_fr = vary_fr

#     def construct(self, x,ids_keep=None,**kwargs):
#         # convert to list

#         if not isinstance(x, list):
#             x = [x]

#         if True:
#             if len(x) == 10:
#                 idx_crops = [2,10]
#             elif len(x) == 2:
#                 idx_crops = [2]

#         start_idx = 0
#         for end_idx in idx_crops:
#             _out = self.backbone(P.concat(x[start_idx: end_idx]), ids_keep=ids_keep,**kwargs)
#             if start_idx == 0:
#                 output = _out
#             else:
#                 if isinstance(_out, tuple):
#                     output1 = P.concat((output[0], _out[0]))
#                     output2 = P.concat((output[1], _out[1]))
#                     output = (output1, output2)
#                 else:
#                     output = P.concat((output, _out))
#             start_idx = end_idx

#         return output



# # TODO
# #已改 
# def get_params_groups(model):
#     regularized = []
#     not_regularized = []
#     for param in model.get_parameters():
#         if not param.requires_grad:
#             continue
#         # we do not regularize biases nor Norm parameters
#         if param.name.endswith(".bias") or len(param.shape) == 1:
#             not_regularized.append(param)
#         else:
#             regularized.append(param)
#     return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

# #TODO
# def has_batchnorms(model):
#     bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
#     for name, module in model.cells_and_names():
#         if isinstance(module, bn_types):
#             return True
#     return False

# # TODO
# def get_diff_images(images, idx=None):
#     if idx is None:
#         return [im[:, :, 1:, ...] - im[:, :, :-1, ...] for im in images]

#     else:
#         return [im[:, :, idx + 1, :, :] - im[:, :, idx, :, :] for im in images]

# # TODO
# def get_flow_images(images, temporal_length=8):
#     out_list = []
#     for im in images:
#         idx = np.random.randint(0, temporal_length)
#         out_list.append(im[:, :, idx, ...])
#     return out_list





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
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import mindspore
import os
import sys
from mindspore import Tensor
import mindspore.ops as P
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init, get_group_size, get_rank
import argparse
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
from PIL import ImageFilter, ImageOps
import warnings
from mindspore.nn import Cell
from mindspore import ms_function

# ok
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

# TODO mindspore的梯度不存在tensor中
def clip_gradients(model, clip):
    norms = []
    for name, p in model.parameters_and_names():
        p.grad = None
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

# 已改
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.parameters_and_names():
        if "last_layer" in n:
            p.grad = None

#已改
def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    # checkpoint = torch.load(ckp_path, map_location="cpu")
    checkpoint=mindspore.load_checkpoint(ckp_path)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                # msg = value.load_state_dict(checkpoint[key], strict=False)
                #TODO 不确定
                msg=mindspore.load_param_into_net(checkpoint[key],value)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    # msg = value.load_state_dict(checkpoint[key])
                    msg=mindspore.load_param_into_net(checkpoint[key],value)
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]



# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    schedule=Tensor(schedule,mindspore.float32)
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def sgd_lr(base_value,final_value,epochs,niter_per_ep,warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    lr1=np.linspace(base_value, base_value, niter_per_ep*5)
    schedule = np.concatenate((warmup_schedule,lr1))
    lrt=np.linspace(base_value/2, base_value/2, niter_per_ep*5)  #  15 1e-2
    schedule = np.concatenate((schedule,lrt))
    lr2=np.linspace(base_value/10, base_value/10, niter_per_ep*5)  # 19  1e-3
    schedule = np.concatenate((schedule, lr2))
    lr3=np.linspace(base_value/100, base_value/100, niter_per_ep*5)
    schedule = np.concatenate((schedule, lr3))
    lr4=np.linspace(base_value/1000, base_value/1000, niter_per_ep*(epochs-25))
    schedule = np.concatenate((schedule, lr4))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# def sgd_lr(base_value,final_value,epochs,niter_per_ep,warmup_epochs=0, start_warmup_value=0):
#     # warmup_schedule = np.array([])
#     # warmup_iters = warmup_epochs * niter_per_ep
#     # if warmup_epochs > 0:
#     #     warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     lr1=np.linspace(base_value, base_value, niter_per_ep*5)
#     # schedule = np.concatenate((warmup_schedule,lr1))
#     lrt=np.linspace(base_value/2, base_value/2, niter_per_ep*5)  #  15 1e-3
#     schedule = np.concatenate((lr1,lrt))
#     lr2=np.linspace(base_value/10, base_value/10, niter_per_ep*3)  # 19  1e-4
#     schedule = np.concatenate((schedule, lr2))
#     lr3=np.linspace(base_value/100, base_value/100, niter_per_ep*2)
#     schedule = np.concatenate((schedule, lr3))
#     # lr4=np.linspace(base_value/1000, base_value/1000, niter_per_ep*(epochs-23))
#     # schedule = np.concatenate((schedule, lr4))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule


class cosine_scheduler_weightDecay(Cell):
    def __init__(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        super(cosine_scheduler_weightDecay, self).__init__()
        self.warmup_schedule = np.array([])
        self.warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            self.warmup_schedule = np.linspace(start_warmup_value, base_value, self.warmup_iters)

        iters = np.arange(epochs * niter_per_ep - self.warmup_iters)
        self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((self.warmup_schedule, self.schedule))
        assert len(self.schedule) == epochs * niter_per_ep
        self.schedule = Tensor(self.schedule,mindspore.float32)

    def construct(self, global_step):
        
        return self.schedule[global_step]




zeroslike = P.ZerosLike()
def cancel_gradients_last_layer(epoch, grads, freeze_last_layer):
    new_grads=[]
    if epoch >= freeze_last_layer:
        return grads
    for g in grads:
        if g.shape==(65536,256):
            g = zeroslike(g)  # 将梯度置为None
            new_grads.append(g)
        else:
            new_grads.append(g)
    return tuple(new_grads)





# ok
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


# ok
def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    mindspore.set_seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # if not is_dist_avail_and_initialized():
        #     return
        t = mindspore.Tensor([self.count, self.total], dtype=mindspore.float64)
        t = mindspore.ops.AllReduce()(t)

        t = t.asnumpy()
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = mindspore.Tensor(list(self.deque))
        d = d.asnumpy()
        e = np.median(d)
        return float(e)

    @property
    def avg(self):
        d = mindspore.Tensor(list(self.deque), dtype=mindspore.float32)
        d = d.asnumpy()
        d = np.mean(d)
        return float(d)

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# def reduce_dict(input_dict, average=True):
#     """
#     Args:
#         input_dict (dict): all the values will be reduced
#         average (bool): whether to do average or sum
#     Reduce the values in the dictionary from all processes so that all processes
#     have the averaged results. Returns a dict with the same fields as
#     input_dict, after reduction.
#     """
#     world_size = get_world_size()
#     if world_size < 2:
#         return input_dict
#     with torch.no_grad():
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#         values = torch.stack(values, dim=0)
#         dist.all_reduce(values)
#         if average:
#             values /= world_size
#         reduced_dict = {k: v for k, v in zip(names, values)}
#     return reduced_dict

# TODO
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, mindspore.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable,data_size, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(data_size))) + 'd'
        # if torch.cuda.is_available():
        #     log_msg = self.delimiter.join([
        #         header,
        #         '[{0' + space_fmt + '}/{1}]',
        #         'eta: {eta}',
        #         '{meters}',
        #         'time: {time}',
        #         'data: {data}',
        #         'max mem: {memory:.0f}'
        #     ])
        # else:
        #     log_msg = self.delimiter.join([
        #         header,
        #         '[{0' + space_fmt + '}/{1}]',
        #         'eta: {eta}',
        #         '{meters}',
        #         'time: {time}',
        #         'data: {data}'
        #     ])
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # if torch.cuda.is_available():
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time),
                #         memory=torch.cuda.max_memory_allocated() / MB))
                # else:
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# def get_sha():
#     cwd = os.path.dirname(os.path.abspath(__file__))
#
#     def _run(command):
#         return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
#     sha = 'N/A'
#     diff = "clean"
#     branch = 'N/A'
#     try:
#         sha = _run(['git', 'rev-parse', 'HEAD'])
#         subprocess.check_output(['git', 'diff'], cwd=cwd)
#         diff = _run(['git', 'diff-index', 'HEAD'])
#         diff = "has uncommited changes" if diff else "clean"
#         branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
#     except Exception:
#         pass
#     message = f"sha: {sha}, status: {diff}, branch: {branch}"
#     return message


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()

# TODO
def is_main_process():
    # return get_rank() == 0
    return True

# TODO
def save_on_master(*args, **kwargs):
    if is_main_process():
        mindspore.save_checkpoint(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

MODE = {"PYNATIVE_MODE": context.PYNATIVE_MODE,
        "GRAPH_MODE": context.GRAPH_MODE}
def cloud_context_init(
                        seed=0,
                       use_parallel=True,
                       context_config=None,
                       parallel_mode=None):
    np.random.seed(seed)
    set_seed(seed)
    # mode_config = context.GRAPH_MODE
    context_config["mode"] = MODE[context_config["mode"]]
    rank_id, device_num = 0, 1
    if use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        context_config["device_id"] = device_id
        parallel_mode=context.ParallelMode.DATA_PARALLEL
        context.set_context(**context_config)
        init()
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        context.set_auto_parallel_context(
            device_num=device_num, gradients_mean=True,parallel_mode=parallel_mode)
    os.environ['MOX_SILENT_MODE'] = '1'
    return rank_id, device_num




# def init_distributed_mode(args):
#     # launched with torch.distributed.launch
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     # launched with submitit on a slurm cluster
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     # launched naively with `python main_dino.py`
#     # we manually add MASTER_ADDR and MASTER_PORT to env variables
#     elif torch.cuda.is_available():
#         print('Will run the code on one GPU.')
#         args.rank, args.gpu, args.world_size = 0, 0, 1
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = '29500'
#     else:
#         print('Does not support training without GPU.')
#         sys.exit(1)

#     dist.init_process_group(
#         backend="nccl",
#         init_method=args.dist_url,
#         world_size=args.world_size,
#         rank=args.rank,
#     )

#     torch.cuda.set_device(args.gpu)
#     print('| distributed init (rank {}): {}'.format(
#         args.rank, args.dist_url), flush=True)
#     dist.barrier()
#     setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # Cut & paste from PyTorch official master until it's in a few official releases - RW
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor

#已改
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    #TODO 不确定是不是这样改
    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    #TODO 以下可能会出问题
    size=tensor.shape
    a=Tensor(np.random.uniform(2 * l - 1, 2 * u - 1,size=size),mindspore.float32)
    mindspore.ops.stop_gradient(a)
    #tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal

    # tensor.erfinv_()
    a=P.Erfinv()(a)
    mindspore.ops.stop_gradient(a)

    # Transform to proper mean, std
    a=P.mul(a,std * math.sqrt(2.))
    mindspore.ops.stop_gradient(a)
    a=P.add(a,mean)
    mindspore.ops.stop_gradient(a)
    # tensor.mul_(std * math.sqrt(2.))
    # tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    # tensor.clamp_(min=a, max=b)
    tensor = mindspore.numpy.clip(tensor, xmin=a, xmax=b)
    mindspore.ops.stop_gradient(tensor)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




class Identity(nn.Cell):

    def construct(self, x):
        return x


class MultiCropWrapper(nn.Cell):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, vary_fr=False):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        if hasattr(backbone, 'fc'):
            backbone.fc, backbone.head = Identity(), Identity()
        self.backbone = backbone
        # self.head = head
        self.vary_fr = vary_fr

    def construct(self, x,ids_keep=None,**kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = []
        output=[]
        if True:
            if len(x) == 10:
                idx_crops = [2,10]
            elif len(x) == 2:
                idx_crops = [2]
        start_idx = 0
        for end_idx in idx_crops:
            
            _out = self.backbone(P.concat(x[start_idx: end_idx]), ids_keep=ids_keep,**kwargs)
            if start_idx == 0:
                output = _out
            else:
                if isinstance(_out, tuple):
                    output1 = P.concat((output[0], _out[0]))
                    output2 = P.concat((output[1], _out[1]))
                    output = (output1, output2)
                else:
                    output = P.concat((output, _out))
            start_idx = end_idx
        # return self.head(output)
        return output

#     def construct(self, x,ids_keep=None,**kwargs):
#         # convert to list

#         if not isinstance(x, list):
#             x = [x]

#         if True:
#             if len(x) == 10:
#                 idx_crops = [2,10]
#             elif len(x) == 2:
#                 idx_crops = [2]

#         start_idx = 0
#         for end_idx in idx_crops:
#             _out = self.backbone(P.concat(x[start_idx: end_idx]), ids_keep=ids_keep,**kwargs)
#             if start_idx == 0:
#                 output = _out
#             else:
#                 if isinstance(_out, tuple):
#                     output1 = P.concat((output[0], _out[0]))
#                     output2 = P.concat((output[1], _out[1]))
#                     output = (output1, output2)
#                 else:
#                     output = P.concat((output, _out))
#             start_idx = end_idx

#         return output

    
    
    
    
    

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for param in model.get_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if param.name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.cells_and_names():
        if isinstance(module, bn_types):
            return True
    return False

def get_diff_images(images, idx=None):
    if idx is None:
        return [im[:, :, 1:, ...] - im[:, :, :-1, ...] for im in images]

    else:
        return [im[:, :, idx + 1, :, :] - im[:, :, idx, :, :] for im in images]


def get_flow_images(images, temporal_length=8):
    out_list = []
    for im in images:
        idx = np.random.randint(0, temporal_length)
        out_list.append(im[:, :, idx, ...])
    return out_list