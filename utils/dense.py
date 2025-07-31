from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import constexpr
import mindspore as ms
from mindspore.common.initializer import initializer, TruncatedNormal, Zero, One

def norm_except_dim(v, pow, dim):
    if dim == -1:
        return mnp.norm(v, pow)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return mnp.norm(v.view((v.shape[0], -1)), pow, 1).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return mnp.norm(v.view((-1, v.shape[v.ndim - 1])), pow, 0).view(output_size)
    else:
        return norm_except_dim(v.swapaxes(0, dim), pow, dim).swapaxes(0, dim)

def _weight_norm(v, g, dim):
    return v * (g / norm_except_dim(v, 2, dim))

class dense_(nn.Cell):
    r"""Applies weight normalization to a parameter in the given module.
    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}
    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. 
    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.
    See https://arxiv.org/abs/1602.07868
    Args:
        module (Module): containing module
        dim (int, optional): dimension over which to compute the norm
    Returns:
        The original module with the weight norm hook
    Example::
        >>> m = WeightNorm(nn.Dense(20, 40))
        >>> m.param_g.shape
        (40, 1)
        >>> m.param_v.shape
        (40, 20)
    """
    def __init__(self,input_dim, out_dim):
        super().__init__()

        self.dim = 0
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.ones=ms.ops.Ones()
        self.matmul=ms.ops.MatMul(transpose_b=True)
        self.weight_g = Parameter(self.ones((self.out_dim, 1), ms.float32),requires_grad=False)
        self.weight_v = Parameter(initializer(TruncatedNormal(0.02), [self.out_dim, self.input_dim], ms.float32))
        
    def construct(self, x):
        
        output=self.matmul(x,_weight_norm(self.weight_v, self.weight_g, self.dim))
        
        return output