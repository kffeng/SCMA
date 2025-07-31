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


Assign = P.Assign()

class BYOLLoss(nn.Cell):
    def __init__(self, g_crops=2,l_crops=10):
        super().__init__()

        self.g_crops = g_crops
        self.l_crops = l_crops

    def forward_loss(self,q,k):
        
        loss = ((q - k) ** 2).sum(axis=-1)
        
        return loss.mean()
     
    def construct(self, x_s, x_t):

        teacher_out = ms.ops.split(x_t,axis=0,output_num=self.g_crops)
        student_out = ms.ops.split(x_s,axis=0,output_num=self.l_crops)
        total_loss = 0
        n_loss_terms = 0
        for iq, k in enumerate(teacher_out):
            for q in range(len(student_out)):
                if q == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss=self.forward_loss(student_out[q],k)
                total_loss += loss
                n_loss_terms += 1
        total_loss=total_loss/n_loss_terms
        
        return total_loss
