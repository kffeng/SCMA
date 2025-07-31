import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import mindspore.communication.management as distributed
from mindspore.ops import stop_gradient
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from mindspore.common.initializer import initializer, TruncatedNormal, Zero, One
from utils.utils import trunc_normal_
from utils.weight import WeightNorm
from mindspore import ms_function
from utils.dense import dense_

class ContrastHead_Student(nn.Cell):
    def __init__(self, in_dim, out_dim, hidden_dim=4096):
        super().__init__()

        self.proj = nn.SequentialCell([
            nn.Dense(in_dim, hidden_dim).to_float(ms.float16),
            nn.LayerNorm((hidden_dim,)),
            nn.GELU(),
            nn.Dense(hidden_dim, hidden_dim).to_float(ms.float16),
            nn.LayerNorm((hidden_dim,)),
            nn.GELU(),
            nn.Dense(hidden_dim, out_dim).to_float(ms.float16),
        ])

        self.pred = nn.SequentialCell([
            nn.Dense(out_dim, hidden_dim).to_float(ms.float16),
            nn.LayerNorm((hidden_dim,)),
            nn.GELU(),
            nn.Dense(hidden_dim, out_dim).to_float(ms.float16),
        ])

        self.norm = ops.L2Normalize(axis=-1)
        self.init_weights()


    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(0.0, cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Zero(), cell.beta.shape, cell.beta.dtype))


    def construct(self, x):
        x = self.proj(x)
        x = self.pred(x)
        x = self.norm(x)

        return x


class ContrastHead_Teacher(nn.Cell):
    def __init__(self, in_dim, out_dim, hidden_dim=4096):
        super().__init__()

        self.proj = nn.SequentialCell([
            nn.Dense(in_dim, hidden_dim).to_float(ms.float16),
            nn.LayerNorm((hidden_dim,)),
            nn.GELU(),
            nn.Dense(hidden_dim, hidden_dim).to_float(ms.float16),
            nn.LayerNorm((hidden_dim,)),
            nn.GELU(),
            nn.Dense(hidden_dim, out_dim).to_float(ms.float16),
        ])


        self.norm = ops.L2Normalize(axis=-1)
        self.init_weights()


    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(0.0, cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Zero(), cell.beta.shape, cell.beta.dtype))


    def construct(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x



class DINOHead(nn.Cell):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Dense(in_dim, bottleneck_dim)
        else:
            layers = [nn.Dense(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU(approximate=False))
            for _ in range(nlayers - 2):
                layers.append(nn.Dense(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU(approximate=False))
            layers.append(nn.Dense(hidden_dim, bottleneck_dim))
            self.mlp = nn.SequentialCell(layers)
        self.init_weights()

        self.last_layer =dense_(bottleneck_dim, out_dim)
        self.norm = P.L2Normalize(-1,epsilon=1e-12)

    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(0.0, cell.bias.shape))

    def construct(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        x = self.last_layer(x)
        return x
