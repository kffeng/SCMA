import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops
from mindspore import Tensor
from mindspore.communication import init, get_group_size, get_rank


class DINOLoss(nn.Cell):
    def __init__(self, args,out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops

        self.args=args
        # self.get_group_size=get_group_size()
        self.log_softmax = ms.nn.LogSoftmax()
        self.op_sum = ms.ops.ReduceSum(keep_dims=False)
        self.softmax=ms.ops.Softmax()
        self.op_sum2= ms.ops.ReduceSum(keep_dims=True)
        self.allreduce = AllReduce()
        
        # self.register_buffer("center", ms.ops.Zeros()((1, out_dim), ms.float32))
        self.center = ms.Parameter(ms.ops.Zeros()((1, out_dim), ms.float32),name='center',requires_grad = False)

        
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def construct(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0

        student_out = student_output / self.student_temp
        student_out = ms.ops.split(student_out,axis=0,output_num=self.n_crops)
        
        temp=Tensor(0.04)
        teacher_out = self.softmax((teacher_output - self.center) / temp)
        for v in range(len(student_out)):
            loss=self.op_sum(-teacher_out*self.log_softmax(student_out[v]),-1)
            total_loss += ms.ops.mean(loss)
            n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        if isinstance(teacher_output, (tuple, list)):
            batch_center = ms.ops.stop_gradient([ms.ops.ReduceSum(keep_dims=True)(x, axis=0) for x in teacher_output])
            batch_center[0] = ms.ops.AllReduce()(batch_center[0])
            batch_center[1] = ms.ops.AllReduce()(batch_center[1])
            batch_center = ms.ops.stop_gradient([x / (len(teacher_output[0]) * 8) for x in batch_center])
            self.center[0, :] = self.center[0, :] * self.center_momentum + batch_center[0] * (1 - self.center_momentum)
            self.center[1, :] = self.center[1, :] * self.center_momentum + batch_center[1] * (1 - self.center_momentum)
        else:
            batch_center = self.op_sum2(teacher_output, 0)
            batch_center = self.allreduce(batch_center)
            batch_center = batch_center / (len(teacher_output)*4*8)  # 1是节点数
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        return 1

class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.allreduce = ms.ops.AllReduce()
    def construct(self, x):
        return self.allreduce(x)