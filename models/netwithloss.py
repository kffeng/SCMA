import mindspore.numpy as msnp
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import mindspore.communication.management as distributed
from mindspore.ops import stop_gradient
from mindspore import ms_function


class NetWithLossCell(nn.Cell):
    def __init__(self, student, teacher,dino_loss,contrast_loss,data_size,args):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.dino_loss=dino_loss
        self.args=args
        self.contrast_loss=contrast_loss
        self.data_size=data_size

    def construct(self, g_images, l_images, ids_keep=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        g_images=g_images.transpose(1,0,2,3,4,5)
        l_images=l_images.transpose(1,0,2,3,4,5)

        g_images=list(msnp.split(g_images,g_images.shape[0],axis=0))
        l_images=list(msnp.split(l_images,l_images.shape[0],axis=0))


        images_g=[]
        images_l=[]
        for i, x in enumerate(g_images):
            images_g.append(x[0,:,:,:,:,:])
        for i, x in enumerate(l_images):
            images_l.append(x[0,:,:,:,:,:])
        images=images_g+images_l


        t_con_out,t_dino_out = self.teacher(images[:2])
        s_con_out,s_dino_out = self.student(images,ids_keep)

        t_con_out=stop_gradient(t_con_out)
        t_dino_out=stop_gradient(t_dino_out)

        dino_loss=self.dino_loss(s_dino_out,t_dino_out)
        contrast_loss=self.contrast_loss(s_con_out,t_con_out)
        
        print(contrast_loss)
 
        loss=dino_loss+contrast_loss
        # loss=dino_loss

        return loss