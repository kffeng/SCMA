import mindspore.numpy as msnp
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import mindspore.communication.management as distributed
from mindspore.ops import stop_gradient
from mindspore import ms_function
from mindspore import Tensor
import mindspore



def cosine_similarity(x1, x2, dim=1, eps=1e-08):
 
    sum_=mindspore.ops.ReduceSum()
    molecule = sum_(x1 * x2, 1)
    denominator = (ops.norm(x1, axis=dim, p=2) * ops.norm(x2, axis=dim, p=2))
    denominator=ops.clip_by_value(denominator,clip_value_min=Tensor(eps,ms.float32))
    output = molecule / denominator
    return output



class NetWithLossCell(nn.Cell):
    def __init__(self, student, teacher,dino_loss,reco_loss,data_size,local_rank,args):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.dino_loss=dino_loss
        self.args=args
        self.reco_loss=nn.SmoothL1Loss(reduction='mean')
        #self.reco_loss=nn.MSELoss()
        self.local_rank=local_rank
        # selfs.byol_loss=byol_loss
        self.data_size=data_size
        self.cast = ops.Cast()


        
    def construct(self, g_images, l_images, ids_keep_g=None, ids_mask_g=None, ids_keep_l=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        l_images=l_images.transpose(1,0,2,3,4,5)  #(2, 8, 3, 8, 224, 224)
        l_images=list(msnp.split(l_images,l_images.shape[0],axis=0))
        
        B,C,T,H,W = g_images.shape  #(2, 3, 16, 224, 224)

        images_g=[g_images]
        images_l=[]
        for i, x in enumerate(l_images):
            images_l.append(x[0,:,:,:,:,:])
        images=images_g+images_l
        
        t_dino_out, x_full = self.teacher(images[0])
        t_dino_out=stop_gradient(t_dino_out)
        x_full=stop_gradient(x_full)

        ids_keep_g=ops.tile(ids_keep_g,(1,T,1))
        ids_mask_g=ops.tile(ids_mask_g,(1,T,1))
        
        Bl,Tl,V=ids_keep_l.shape
        ids_keep_l=ops.expand_dims(ids_keep_l,2)
        ids_keep_l=msnp.tile(ids_keep_l,(1, 1, 8, 1))
        ids_keep_l=ids_keep_l.transpose(1,0,2,3).reshape(-1,V) #8*2*8,49   8duan*bs*t
        
        s_dino_out, decoder_x= self.student(images,ids_keep_g,ids_mask_g,ids_keep_l)
        B1,_,C1=x_full.shape
        Bm,Tm,Lm=ids_mask_g.shape 
        ids_mask_g=ids_mask_g.reshape(-1,Lm)

        x_mask = x_full.gather_elements(dim=1, index=ids_mask_g.expand_dims(-1).repeat(C1, axis=-1)).view(B1,-1,C1).reshape(B,-1,C1)

        # reco_loss = 1 - cosine_similarity(x_mask, decoder_x).mean()
        reco_loss=self.reco_loss(decoder_x,x_mask).mean()
        if self.local_rank in [0,8,16,24,32,40,48,56]:
            print(f'reco_loss:{reco_loss}')
        dino_loss=self.dino_loss(s_dino_out,t_dino_out)

        loss=dino_loss+reco_loss

        return loss