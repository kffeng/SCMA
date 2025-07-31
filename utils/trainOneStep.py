# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""misc functions for program"""
import mindspore as ms
from mindspore import nn
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from utils.ema import EMACell
from mindspore.common.parameter import ParameterTuple
from utils.utils import cancel_gradients_last_layer

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


class TrainOneStepWithLossScaleCellGlobalNormClip(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, teacher,student,optimizer,momentum_schedule,data_size,args,
                 scale_sense=1.0, use_global_norm=True,
                 clip_global_norm_value=1.0,ema=True):
        super(TrainOneStepWithLossScaleCellGlobalNormClip, self).__init__(network,optimizer,scale_sense)
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.teacher=teacher
        self.student=student
        self.momentum_schedule=momentum_schedule
        self.args=args
        self.ema=ema
        self.data_size=data_size
        self.counter = ms.Parameter(ms.Tensor(0, ms.int64), 'counter_',requires_grad=False)
        # self.ema_model = EMACell(self.teacher,
        #                          ema_decay=self.momentum_schedule[self.counter])
        self.print = P.Print()

        if self.ema:
            self.student_param=list()
            self.teacher_param=list()
            self.assign=ms.ops.Assign()
            self.get_param()

    def get_param(self):
        for param in self.student.get_parameters():
            self.student_param.append(param)
        for param in self.teacher.get_parameters():
            self.teacher_param.append(param)
        self.student_param=ParameterTuple(self.student_param)
        self.teacher_param=ParameterTuple(self.teacher_param)

    def ema_updata(self):
        if self.ema:
            m=self.momentum_schedule[self.counter]
            for s_p, t_p in zip(self.student_param,self.teacher_param):
                self.assign(t_p,t_p*m+s_p*(1-m))
        return 1

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            grads=cancel_gradients_last_layer(self.counter//self.data_size, grads, 1)            # if self.args.local_rank==0:
            loss = F.depend(loss, self.optimizer(grads))
            self.ema_updata()
            # self.ema_model(weights, self.momentum_schedule[self.counter])
        else:
            self.print("=============Over Flow, skipping=============")
        ms.ops.assign_add(self.counter, ms.Tensor(1, ms.int64))
        return loss


def get_train_one_step(args, net_with_loss,teacher ,student,optimizer,momentum_schedule,data_size):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        args.logger.info(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 16, scale_factor=2,
                                                                    scale_window=2000)
    else:
        args.logger.info(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    net_with_loss = TrainOneStepWithLossScaleCellGlobalNormClip(net_with_loss, teacher,student,optimizer, momentum_schedule,data_size,args,scale_sense=scale_sense,
                                  clip_global_norm_value=args.clip_global_norm_value,
                                  use_global_norm=True)
    return net_with_loss



class TrainOneStepWithLossScaleCellGlobalNormClip_finetune(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network,optimizer,args,scale_sense=1.0, use_global_norm=True,
                 clip_global_norm_value=1.0):
        super(TrainOneStepWithLossScaleCellGlobalNormClip_finetune, self).__init__(network,optimizer,scale_sense)
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.args=args
        self.print = P.Print()

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            self.print("=============Over Flow, skipping=============")
        return loss


def get_finetune_one_step(args, net_with_loss,optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        args.logger.info(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 16, scale_factor=2,
                                                                    scale_window=2000)
    else:
        args.logger.info(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    net_with_loss = TrainOneStepWithLossScaleCellGlobalNormClip_finetune(net_with_loss,optimizer,args,scale_sense=scale_sense,
                                  clip_global_norm_value=args.clip_global_norm_value,
                                  use_global_norm=True)
    return net_with_loss