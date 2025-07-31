# Copyright 2022 Huawei Technologies Co., Ltd
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
"""ema define"""

import mindspore.nn as nn
from mindspore import Tensor,Parameter
from mindspore import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.ops
from mindspore.common.parameter import ParameterTuple
import numpy as np
from mindspore import ms_function

_ema_op = C.MultitypeFuncGraph("grad_ema_op")
Assign = P.Assign()
AssignAdd = P.AssignAdd()
cast=mindspore.ops.Cast()


def get_param_groups(network):
    """ get param groups """
    decay_params = []
    no_decay_params = []
    for key, param in network.parameters_and_names():
        # parameter_name = x.name
        if key.endswith(".moving_mean") or key.endswith(".moving_variance") :
            continue
        if (key.endswith(".gamma") and param.shape == (256,)) or (key.endswith(".beta") and param.shape == (256,)):
            continue
        elif key.endswith(".bias") or len(param.shape) == 1:
            # Dense or Conv's weight using weight decay
            no_decay_params.append(param)
        elif param.shape != (65536,1):
            # all bias not using weight decay
            # bn weight bias not using weight decay, be carefully for now x not include LN
            decay_params.append(param)

    return no_decay_params+decay_params





@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    # temp=Tensor(np.zeros([1]).astype(np.float32))
    # temp=mindspore.ops.fill(mindspore.float32,ema_weight.shape,0)
    """Apply grad sum to cumulative gradient."""
    # print("self.ema_decay:{}".format(factor))
    # ema_weight=cast(ema_weight,mstype.float32)
    # if ema_weight.shape==(65536,1):
    #     return Assign(ema_weight, ema_weight)
    # print("leicing1:{}".format(type(ema_weight)))
    # print("leicing2:{}".format(type(weight)))
    # print("changdu1:{}".format(len(ema_weight)))
    # print("changdu2:{}".format(len(weight)))
    # print("zheshi1:{}".format(ema_weight.shape))
    # print("zheshi2:{}".format(weight.shape))
    # print("laosh:{}".format(ema_weight[0]))
    # print("xuesheng:{}".format(weight[0]))
    # ema_weight=Parameter(ema_weight,requires_grad=False)

    return Assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class EMACell(nn.Cell):
    """EMACell Define"""
    def __init__(self, weights, ema_decay=0.9999):
        super(EMACell, self).__init__()
        # self.ema_weights = weights.clone(prefix="_ema_weights")
        self.ema_decay = Parameter(Tensor(ema_decay, mstype.float32))
        # self.teacher = ParameterTuple(weights.parameters_dict().values())
        self.teacher=ParameterTuple(get_param_groups(weights))
        
        self.hyper_map = C.HyperMap()
    # @ms_function
    def construct(self, weights,ema_decay):
        # self.ema_decay=ema_decay
        # self.teacher=teacher
        
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay), self.teacher, weights)
        # success=mindspore.ops.stop_gradient(success)
        return success