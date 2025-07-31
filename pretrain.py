import argparse
import os

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor
from optim.build_optim import get_optimizer
from models.timesformerv2 import get_vit_base_patch16_224
from models.dinoloss import DINOLoss
from models.contrastloss import ContrastLoss
from models.rec_loss import Reco_loss
from models.netwithlossv2 import NetWithLossCell
from models.vision_transformer import DINOHead
from models.build_network import build_student,build_teacher
from models.decoder import VisionTransformerDecoder

from utils.trainOneStep import get_train_one_step
from datasets.video_datasetv2 import create_dataset
from utils.config_parser import get_config
from utils.helper import cloud_context_init
from utils.logger import get_logger
from utils.utilsv2 import MultiCropWrapper_Student,MultiCropWrapper_Teacher
from utils.utilsv2 import cosine_scheduler,cosine_scheduler_weightDecay
from utils.MyLossMonitor import LossMonitorsvt,StopAtStep


def main(args):
    context_config = {
        "mode": args.mode,
        "device_target": args.device_target,
        "device_id": args.device_id,
        'max_call_depth': args.max_call_depth,
        'save_graphs': args.save_graphs,
    }
    parallel_config = {
        'parallel_mode': args.parallel_mode,
        'gradients_mean': args.gradients_mean,
    }
    local_rank, device_id, device_num = cloud_context_init(seed=args.seed,
                                                           use_parallel=args.use_parallel,
                                                           context_config=context_config,
                                                           parallel_config=parallel_config)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info(f"local_rank: {local_rank}, device_num: {device_num}, device_id: {device_id}")

    # train dataset
    dataset = create_dataset(args, mode="pretrain", shuffle=True)

    data_size = dataset.get_dataset_size()
    new_epochs = (args.epochs - args.epochstart)
    if args.per_step_size > 0:
        new_epochs = int((data_size / args.per_step_size) * (args.epochs - args.epochstart))
    elif args.per_step_size == 0:
        args.per_step_size = data_size
    args.logger.info("Will be Training epochs:{}, sink_size:{}".format(new_epochs, args.per_step_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    student = get_vit_base_patch16_224(cfg=args, no_head=True,masked_im_modeling=args.masked_im_modeling)
    teacher = get_vit_base_patch16_224(cfg=args, no_head=True)
    
    size = ops.Size()
    n_parameters = sum(size(p) for p in student.trainable_params() if p.requires_grad)
    args.logger.info("number of params: {}".format(n_parameters))

    embed_dim = student.embed_dim
    embed_dim = student.embed_dim
    student = MultiCropWrapper_Student(student, vary_fr=args.RAND_FR)
    teacher = MultiCropWrapper_Teacher(teacher,vary_fr=args.RAND_FR)

    # define lr_schedule
    momentum_schedule=cosine_scheduler(args.momentum_teacher,1,args.epochs,data_size)
    lr_schedule =cosine_scheduler(args.start_learning_rate,args.end_learning_rate,args.epochs,data_size,warmup_epochs=args.warmup_epochs)
    lr_schedule=Tensor(lr_schedule)

    #define head
    s_dinohead=DINOHead(embed_dim,args.out_dim,args.use_bn_in_head,args.norm_last_layer)
    t_dinohead=DINOHead(embed_dim,args.out_dim,args.use_bn_in_head,args.norm_last_layer)
    cro_att=VisionTransformerDecoder(embed_dim,num_heads=12,qkv_bias=True)

    student=build_student(student,cro_att,s_dinohead)
    teacher=build_teacher(teacher,t_dinohead)

    # define optimizer
    optimizer = get_optimizer(args, student, data_size)

    # define dino_loss
    dino_loss = DINOLoss(args,args.out_dim, args.local_crops_number, 
                         args.warmup_teacher_temp, args.teacher_temp,
                         args.warmup_teacher_temp_epochs, args.epochs, global_crops=2)
    reco_loss=Reco_loss()

    for p in teacher.get_parameters():
        p.requires_grad = False
    args.logger.info("Student and Teacher are built: they are both {} network.".format(args.arch))


    train_net=NetWithLossCell(student,teacher,dino_loss,reco_loss,data_size,args)

    # profile_call_back = StopAtStep(50, 80)
    # load pretrain ckpt 
    if args.use_ckpt:
        args.logger.info(f"Load ckpt: {args.use_ckpt}...")
        params_dict = load_checkpoint(args.use_ckpt)
        msg = load_param_into_net(model, params_dict)
        if len(msg):
            args.logger.info(msg)
        else:
            args.logger.info("All keys match successfully!")

    if args.device_target in ["CPU"]:
        train_model = nn.TrainOneStepCell(student, optimizer)
    else:
        train_model = get_train_one_step(args, train_net, teacher,student ,optimizer,momentum_schedule,data_size)

    callback = [LossMonitorsvt(ifeval=False,log=args.logger)]

    # define ckpt config
    save_ckpt_feq = args.save_ckpt_epochs * args.per_step_size
    if local_rank == 0:
        config_ck1 = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq, keep_checkpoint_max=30,saved_network =teacher,integrated_save=False,
                                     async_save=True, exception_save=True)  
        ckpoint_cb1 = ModelCheckpoint(prefix=args.prefix1, directory=args.save_dir, config=config_ck1)

        config_ck2 = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq, keep_checkpoint_max=30,saved_network = student,integrated_save=False,
                                     async_save=True, exception_save=True)  
        ckpoint_cb2 = ModelCheckpoint(prefix=args.prefix2, directory=args.save_dir, config=config_ck2)
        callback += [ckpoint_cb1, ckpoint_cb2]


    # define Model and begin training
    args.logger.info("Training begin...")
    model = Model(train_model)
    model.train(new_epochs, dataset, callbacks=callback, dataset_sink_mode=args.sink_mode) # , sink_size=args.per_step_size
    # profiler.analyse()
    args.logger.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVT pre-training", add_help=False)
    parser.add_argument('--config_file', type=str, default="../models/configs/Kinetics/TimeSformer_divST_8x32_224_pretrain.yaml") # default_config_ViT-B

    args = parser.parse_args()
    args = get_config(args.config_file)

    main(args)
