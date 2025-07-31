# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Functions of optimizer"""
import os
from mindspore.nn.optim import AdamWeightDecay
import mindspore.nn as nn
from utils.utils import cosine_scheduler,cosine_scheduler_weightDecay,sgd_lr


def get_learning_rate(args,batch_num,mode):
    """Get learning rate"""
    if mode=="adamw":
        learning_rate=cosine_scheduler(args.start_learning_rate*(args.batch_size*4*8/256),args.end_learning_rate,args.epochs,batch_num,args.warmup_epochs)
    else:
        learning_rate=sgd_lr(args.start_learning_rate,args.end_learning_rate,args.epochs,batch_num,args.warmup_epochs)
    return learning_rate


def get_optimizer(args, model, batch_num):
    """Get optimizer for training"""
    args.logger.info(f"=> When using train_wrapper, using optimizer {args.optimizer}")
    args.start_epoch = int(args.epochstart)
    optim_type = args.optimizer.lower()
    params = get_param_groups(model)
    learning_rate = get_learning_rate(args, batch_num,optim_type)
    step = int(args.start_epoch * batch_num)
    accumulation_step = int(args.accumulation_step)
    learning_rate = learning_rate[step::accumulation_step]
    train_step = len(learning_rate)
    weight_decay=cosine_scheduler_weightDecay(args.weight_decay,args.weight_decay_end,args.epochs,batch_num)
    device_num=int(os.getenv("DEVICE_NUM", args.device_num))
    args.logger.info(f"=> Get LR from epoch: {args.start_epoch}\n"
                    f"=> Start step: {step}\n"
                    f"=> Total step: {train_step}\n"
                    f"=> Accumulation step:{accumulation_step}\n"
                    f"=> device_num: {device_num}")
    
    if optim_type == "sgd":
        optim = nn.SGD(
            params=params,
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    elif optim_type == "adamw":
        optim = AdamWeightDecay(
            params=params,
            learning_rate=learning_rate,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"optimizer {optim_type} is not supported")

    return optim


def get_param_groups(network):
    """ get param groups """
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith(".bias") or len(x.shape) == 1:
            # Dense or Conv's weight using weight decay
            no_decay_params.append(x)
        else:
            # all bias not using weight decay
            # bn weight bias not using weight decay, be carefully for now x not include LN
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

