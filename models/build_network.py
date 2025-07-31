import mindspore as ms
from mindspore import nn
from mindspore.ops import stop_gradient,Concat
from mindspore import ms_function

class build_student(nn.Cell):
    def __init__(self, student, cross_att , dinohead):
        super(build_student, self).__init__()
        self.student = student
        self.dinohead = dinohead
        self.cro_att=cross_att
        self.cat=Concat()
        
        
    def construct(self, x_s,ids_keep_g,ids_mask_g,ids_keep_l):
        output_cls, output_vis = self.student(x_s,ids_keep_g,ids_mask_g,ids_keep_l)  # 768
        decoder_x=self.cro_att(output_vis,ids_keep_g,ids_mask_g)
        s_dino_out = self.dinohead(output_cls)  # 65536
        return s_dino_out,decoder_x


class build_teacher(nn.Cell):
    def __init__(self, teacher, dinohead):
        super(build_teacher, self).__init__()
        self.teacher = teacher
        self.dinohead = dinohead
        self.cat=Concat()
        

    def construct(self, x_t):
        output_cls, output_mask = self.teacher(x_t)
        t_dino_out = self.dinohead(output_cls)
        return t_dino_out, output_mask
