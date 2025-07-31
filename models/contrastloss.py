# import mindspore as ms
# import mindspore.nn as nn
# from mindspore import Tensor, Parameter
# import mindspore.ops as ops
# import mindspore.communication.management as distributed
# from mindspore.ops import stop_gradient
# from mindspore.common import dtype as mstype
# from mindspore.ops import operations as P
# from mindspore import ms_function
# import numpy as np
# Assign = P.Assign()
# class ContrastLoss(nn.Cell):
#     def __init__(self, dim=256, K=16384, T=0.1):
#         super().__init__()

#         self.dim = dim
#         self.K = K
#         self.T = T

#         self.con_loss = nn.CrossEntropyLoss()
#         self.matmul = ops.MatMul()
#         self.transpose = ops.Transpose()
#         self.concat = ops.Concat(axis=1)
#         self.stdnormal = ops.StandardNormal(seed=2)
#         self.zero = ops.Zeros()
#         self.norm = ops.L2Normalize(axis=1)
#         self.norm2 = ops.L2Normalize(axis=1)
#         self.queue = Parameter(self.norm2(Tensor(self.stdnormal((self.K, self.dim)), ms.float16)), name="queue", requires_grad=False)
#         self.slide = Tensor(np.arange(0, self.K, 1), mstype.int32)
#         self.queue_ptr = Parameter(Tensor(0, dtype=ms.int64), name="queue_ptr", requires_grad=False)
#         self.allgather = AllGather()
#         self.reducesum = P.ReduceSum()
#         self.mul = P.Mul()
#         self.transpose = ops.Transpose()
#         self.cast = ops.Cast()

#     def _dequeue_and_enqueue(self, keys):
#         """
#         Dequeue and enqueue.
#         """

#         keys = self.allgather(keys)
#         keys=stop_gradient(keys)
#         batch_size = keys.shape[0]
 
#         assert self.K % batch_size == 0  # for simplicity
#         slide=self.slide
#         mask = ops.logical_and((slide >= self.queue_ptr), (slide < (self.queue_ptr + batch_size)))
#         slide = self.cast(ops.nonzero(mask * (slide + 1)).squeeze(), mstype.int32)
#         ops.scatter_update(self.queue, slide, keys)
#         Assign(self.queue_ptr, (self.queue_ptr + batch_size) % self.K)
#         return 1

#     def construct(self, x_s, x_t):
#         # compute logits
#         # positive logits: Nx1
#         # print("contrast")
#         # print(x_s.shape)    #(20, 128)
#         # print(x_t.shape)    #(4, 128)
#         teacher_out = ms.ops.split(x_t,axis=0,output_num=2)
#         student_out = ms.ops.split(x_s,axis=0,output_num=10)
#         total_l_pos=0
#         total_num=0
#         for iq, k in enumerate(teacher_out):
#             for q in range(len(student_out)):
#                 if q == iq:
#                     # we skip cases where student and teacher operate on the same view
#                     continue
#                 l_pos = ms.ops.expand_dims(self.reducesum(self.mul(student_out[q],k), -1), -1)
#                 total_l_pos += l_pos
#                 total_num+=1
#         l_pos=total_l_pos/total_num
        
#         total_l_neg=0
#         total_num=0
        
#         for iq, q in enumerate(student_out):
#                 l_neg = self.matmul(q, self.queue.transpose(1,0))
#                 total_l_neg += l_neg
#                 total_num+=1
#         l_neg=total_l_neg/total_num

#         # negative logits: NxK
#         # logits: Nx(1+K)
#         logits = self.concat((l_pos, l_neg))

#         # apply temperature
#         logits /= self.T

#         # labels: positive key indicators
#         labels = Tensor(self.zero(4, mstype.int32))
        
#         self._dequeue_and_enqueue(teacher_out[0])

#         loss = self.con_loss(logits, labels)

#         return loss


# class AllGather(nn.Cell):
#     def __init__(self):
#         super(AllGather, self).__init__()
#         self.allgather = ops.AllGather()
#     # @ms_function
#     def construct(self, x):
#         out = self.allgather(x)
#         return out





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
class ContrastLoss(nn.Cell):
    def __init__(self, dim=256, K=8192, T=0.1):
        super().__init__()

        self.dim = dim
        self.K = K
        self.T = T

        self.con_loss = nn.CrossEntropyLoss()
        self.matmul = ops.MatMul()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=1)
        self.stdnormal = ops.StandardNormal(seed=2)
        self.zero = ops.Zeros()
        self.norm = ops.L2Normalize(axis=1)
        self.norm2 = ops.L2Normalize(axis=1)
        self.queue = Parameter(self.norm2(Tensor(self.stdnormal((self.K, self.dim)), ms.float16)), name="queue", requires_grad=False)
        self.slide = Tensor(np.arange(0, self.K, 1), mstype.int32)
        self.queue_ptr = Parameter(Tensor(0, dtype=ms.int64), name="queue_ptr", requires_grad=False)
        self.allgather = AllGather()
        self.reducesum = P.ReduceSum()
        self.mul = P.Mul()
        self.transpose = ops.Transpose()
        self.cast = ops.Cast()

    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue and enqueue.
        """

        keys = self.allgather(keys)
        keys=stop_gradient(keys)
        batch_size = keys.shape[0]
 
        assert self.K % batch_size == 0  # for simplicity
        slide=self.slide
        mask = ops.logical_and((slide >= self.queue_ptr), (slide < (self.queue_ptr + batch_size)))
        slide = self.cast(ops.nonzero(mask * (slide + 1)).squeeze(), mstype.int32)
        ops.scatter_update(self.queue, slide, keys)
        Assign(self.queue_ptr, (self.queue_ptr + batch_size) % self.K)
        return 1

    def forward_loss(self,q,k):
        
        l_pos = ms.ops.expand_dims(self.reducesum(self.mul(q, k), -1), -1)
        l_neg = self.matmul(q, self.queue.transpose(1,0))
        logits = self.concat((l_pos, l_neg))
        logits /= self.T
        labels = Tensor(self.zero(4, mstype.int32))
        
        loss = self.con_loss(logits, labels)
        
        return loss
    
    def construct(self, x_s, x_t):
        # compute logits
        # positive logits: Nx1
        # print("contrast")
        # print(x_s.shape)    #(20, 128)
        # print(x_t.shape)    #(4, 128)
        teacher_out = ms.ops.split(x_t,axis=0,output_num=2)
        student_out = ms.ops.split(x_s,axis=0,output_num=10)
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
        
        self._dequeue_and_enqueue(teacher_out[0])

        return total_loss


class AllGather(nn.Cell):
    def __init__(self):
        super(AllGather, self).__init__()
        self.allgather = ops.AllGather()
    # @ms_function
    def construct(self, x):
        out = self.allgather(x)
        return out