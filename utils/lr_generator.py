import numpy as np
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR


class LearningRate(LearningRateSchedule):
    """ LearningRate """
    def __init__(self,
                 start_learning_rate,
                 end_learning_rate,
                 total_epochs,
                 warmup_epochs,
                 steps_per_epoch,
                 power=2.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        decay_steps = total_steps - warmup_steps
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(start_learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(start_learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, start_learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """ construct """
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr
    
    
    
    
"""learning rate generator"""
import math
import numpy as np


def _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps):
    """
    Applies three steps decay to generate learning rate array.
    """
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
        lr_each_step.append(lr)
    return lr_each_step


def _generate_poly_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    """
    Applies polynomial decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max) * base * base
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)
    return lr_each_step


def _generate_cosine_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = lr_max * decayed
        lr_each_step.append(lr)
    return lr_each_step


def _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    """
    Applies liner decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr)
    return lr_each_step



def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, poly, cosine or liner(default)

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    if lr_decay_mode == 'steps':
        lr_each_step = _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps)
    elif lr_decay_mode == 'poly':
        lr_each_step = _generate_poly_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)
    elif lr_decay_mode == 'cosine':
        lr_each_step = _generate_cosine_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)
    else:
        lr_each_step = _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch=120, global_step=0):
    """
    generate learning rate array with cosine

    Args:
       lr(float): base learning rate
       steps_per_epoch(int): steps size of one epoch
       warmup_epochs(int): number of warmup epochs
       max_epoch(int): total epochs of training
       global_step(int): the current start index of lr array
    Returns:
       np.array, learning rate array
    """
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps

    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = base_lr * decayed
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[global_step:]
    return learning_rate
