# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import yaml

# from utils.defaults2 import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    # FIXME 此处修改了
    return parser


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        # yaml_cfg = parse_yaml(args.cfg_file)
        # cfg = args+ yaml_cfg
        # print(type(args))
        cfg.merge_from_file(args.cfg_file)   #通过 merge_from_file 方法，将默认配置中的超参用 yaml 文件中指定的超参进行覆盖
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):     #hasattr() 函数用于判断对象是否包含对应的属性
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    cfg.PATH_TO_DATA_DIR = args.data_path
    cfg.NUM_ENSEMBLE_VIEWS = args.NUM_ENSEMBLE_VIEWS
    cfg.NUM_SPATIAL_CROPS = args.NUM_SPATIAL_CROPS
    cfg.PATH_LABEL_SEPARATOR = args.PATH_LABEL_SEPARATOR
    cfg.PATH_PREFIX = args.PATH_PREFIX
    cfg.TRAIN_JITTER_SCALES = [256, 320]
    cfg.TRAIN_CROP_SIZE = args.TRAIN_CROP_SIZE
    cfg.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]
    cfg.DEFAULT_S = args.DEFAULT_S
    cfg.TEST_CROP_SIZE = args.TEST_CROP_SIZE
    cfg.LONG_CYCLE_SAMPLING_RATE = args.LONG_CYCLE_SAMPLING_RATE
    cfg.SAMPLING_RATE = args.SAMPLING_RATE
    cfg.ENABLE_MULTI_THREAD_DECODE = args.ENABLE_MULTI_THREAD_DECODE
    cfg.DECODING_BACKEND = args.DECODING_BACKEND
    cfg.NUM_FRAMES = args.NUM_FRAMES
    cfg.TARGET_FPS = args.TARGET_FPS
    cfg.NO_RGB_AUG = args.NO_RGB_AUG
    cfg.TWO_TOKEN = args.TWO_TOKEN
    return cfg