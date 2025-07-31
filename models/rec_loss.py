import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import mindspore.communication.management as distributed
from mindspore.ops import stop_gradient
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore import ms_function
import numpy as np


class Reco_loss(nn.Cell):
    def __init__(self):
        super().__init__()
        
        self.loss=nn.SmoothL1Loss(reduction='mean')
     
    def construct(self, x, target):

        total_loss=0
        for i in range(2):
            loss=self.loss(x[i],target[i])
            total_loss+=loss.mean()
        total_loss=total_loss/2.0
        return total_loss
