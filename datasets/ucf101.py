# import os
# import random
# import warnings


# from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
# from datasets.decoder import decode
# from datasets.video_cv2 import get_video_container
# from datasets.transform import VideoDataAugmentationDINO
# import mindspore.dataset.transforms as C2
# import os
# import random
# import warnings

# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

# import mindspore
# import mindspore.dataset
# import numpy as np
# import time
# from datasets.transform import resize
# from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
# from datasets.decoder import decode
# from datasets.video_cv2 import get_video_container
# from datasets.transform import VideoDataAugmentationDINO
# from mindspore.dataset.transforms import TypeCast
# from mindspore.common import Tensor
# from mindspore.ops.primitive import constexpr
# from mindspore._checkparam import Validator as validator
# from mindspore.ops import operations as P
# import mindspore.numpy as msnp
# import multiprocessing
# from mindspore.dataset import DistributedSampler,GeneratorDataset,transforms
# import cv2
# import mindspore.common.dtype as mstype
# from mindspore.dataset.vision import Inter
# import multiprocessing
# import math
# from mindspore.ops import Map
# from mindspore.dataset.vision import Resize,Inter
# from datasets.transformers_utils import GlobalTransform, LocalTransform, Change,Spatial_sampling,TensorNormalize



# class UCF101:
#     """
#     UCF101 video loader. Construct the UCF101 video loader, then sample
#     clips from the videos. For training and validation, a single clip is
#     randomly sampled from every video with random cropping, scaling, and
#     flipping. For testing, multiple clips are uniformaly sampled from every
#     video with uniform cropping. For uniform cropping, we take the left, center,
#     and right crop if the width is larger than height, or take top, center, and
#     bottom crop if the height is larger than the width.
#     """

#     def __init__(self, cfg, mode, num_retries=10):
#         """
#         Construct the UCF101 video loader with a given csv file. The format of
#         the csv file is:
#         ```
#         path_to_video_1 label_1
#         path_to_video_2 label_2
#         ...
#         path_to_video_N label_N
#         ```
#         Args:
#             cfg (CfgNode): configs.
#             mode (string): Options includes `train`, `val`, or `test` mode.
#                 For the train mode, the data loader will take data from the
#                 train set, and sample one clip per video. For the val and
#                 test mode, the data loader will take data from relevent set,
#                 and sample multiple clips per video.
#             num_retries (int): number of retries.
#         """
#         # Only support train, val, and test mode.
#         assert mode in ["train", "val", "test"], "Split '{}' not supported for UCF101".format(mode)
#         self.mode = mode
#         self.cfg = cfg
        
#         self._video_meta = {}
#         self._num_retries = num_retries
#         self._split_idx = mode
#         # For training mode, one single clip is sampled from every video. For validation or testing, NUM_ENSEMBLE_VIEWS
#         # clips are sampled from every video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
#         if self.mode in ["train","val"]:
#             self._num_clips = 1
#         elif self.mode in ["test"]:
#             self._num_clips = (
#                     cfg.NUM_ENSEMBLE_VIEWS * cfg.NUM_SPATIAL_CROPS
#             )

#         print("Constructing UCF101 {}...".format(mode))
#         self._construct_loader()

#     def _construct_loader(self):
#         """
#         Construct the video loader.
#         """
#         path_to_file = os.path.join(
#             self.cfg.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
#         )
#         assert os.path.exists(path_to_file), "{} dir not found".format(
#             path_to_file
#         )

#         self._path_to_videos = []
#         self._labels = []
#         self._spatial_temporal_idx = []
#         with open(path_to_file, "r") as f:
#             for clip_idx, path_label in enumerate(f.read().splitlines()):
#                 assert (
#                         len(path_label.split(self.cfg.PATH_LABEL_SEPARATOR))
#                         == 2
#                 )
#                 path, label = path_label.split(
#                     self.cfg.PATH_LABEL_SEPARATOR
#                 )
#                 for idx in range(self._num_clips):  #_num_clips=30
#                     self._path_to_videos.append(
#                         os.path.join(self.cfg.PATH_PREFIX, path)
#                     )
#                     self._labels.append(int(label))
#                     self._spatial_temporal_idx.append(idx)
#                     self._video_meta[clip_idx * self._num_clips + idx] = {}
#         assert (len(self._path_to_videos) > 0), f"Failed to load UCF101 split {self._split_idx} from {path_to_file}"
#         print(f"Constructing UCF101 dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")

#     def __getitem__(self, index):
#         """
#         Given the video index, return the list of frames, label, and video
#         index if the video can be fetched and decoded successfully, otherwise
#         repeatly find a random video that can be decoded as a replacement.
#         Args:
#             index (int): the video index provided by the pytorch sampler.
#         Returns:
#             frames (tensor): the frames of sampled from the video. The dimension
#                 is `channel` x `num frames` x `height` x `width`.
#             label (int): the label of the current video.
#             index (int): if the video provided by pytorch sampler can be
#                 decoded, then return the index of the video. If not, return the
#                 index of the video replacement that can be decoded.
#         """
#         short_cycle_idx = None
#         # When short cycle is used, input index is a tupple.
#         if isinstance(index, tuple):
#             index, short_cycle_idx = index

#         if self.mode in ["train","val"]:
#             # -1 indicates random sampling.
#             temporal_sample_index = -1
#             spatial_sample_index = -1
#             min_scale = self.cfg.TRAIN_JITTER_SCALES_0
#             max_scale = self.cfg.TRAIN_JITTER_SCALES_1
#             crop_size = self.cfg.TRAIN_CROP_SIZE
#             if short_cycle_idx == 0:
#                 crop_size = int(
#                     round(
#                         self.cfg.SHORT_CYCLE_FACTORS_0
#                         * self.cfg.DEFAULT_S
#                     )
#                 )
#             if short_cycle_idx == 1:
#                 crop_size = int(
#                     round(
#                         self.cfg.SHORT_CYCLE_FACTORS_1
#                         * self.cfg.DEFAULT_S
#                     )
#                 )
#             if self.cfg.DEFAULT_S > 0:
#                 # Decreasing the scale is equivalent to using a larger "span"
#                 # in a sampling grid.
#                 min_scale = int(
#                     round(
#                         float(min_scale)
#                         * crop_size
#                         / self.cfg.DEFAULT_S
#                     )
#                 )
#         elif self.mode in ["test"]:
#             temporal_sample_index = (self._spatial_temporal_idx[index] // self.cfg.NUM_SPATIAL_CROPS)
#             # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
#             # center, or right if width is larger than height, and top, middle,
#             # or bottom if height is larger than width.
#             spatial_sample_index = (
#                 (self._spatial_temporal_idx[index] % self.cfg.NUM_SPATIAL_CROPS)
#                 if self.cfg.NUM_SPATIAL_CROPS > 1 else 1
#             )
#             min_scale, max_scale, crop_size = (
#                 [self.cfg.TEST_CROP_SIZE] * 3 if self.cfg.NUM_SPATIAL_CROPS > 1
#                 else [self.cfg.TRAIN_JITTER_SCALES_0] * 2 + [self.cfg.TEST_CROP_SIZE]
#             )
#             # The testing is deterministic and no jitter should be performed.
#             # min_scale, max_scale, and crop_size are expect to be the same.
#             assert len({min_scale, max_scale}) == 1
#         else:
#             raise NotImplementedError(
#                 "Does not support {} mode".format(self.mode)
#             )
#         sampling_rate = get_random_sampling_rate(
#             self.cfg.LONG_CYCLE_SAMPLING_RATE,
#             self.cfg.SAMPLING_RATE,
#         )
#         # Try to decode and sample a clip from a video. If the video can not be
#         # decoded, repeatedly find a random video replacement that can be decoded.
#         for i_try in range(self._num_retries):
#             video_container = None
#             try:
#                 video_container = get_video_container(
#                     self._path_to_videos[index],
#                     self.cfg.ENABLE_MULTI_THREAD_DECODE,
#                     self.cfg.DECODING_BACKEND,
#                 )
#             except Exception as e:
#                 print(
#                     "Failed to load video from {} with error {}".format(
#                         self._path_to_videos[index], e
#                     )
#                 )
#             # Select a random video if the current video was not able to access.
#             if video_container is None:
#                 warnings.warn(
#                     "Failed to meta load video idx {} from {}; trial {}".format(
#                         index, self._path_to_videos[index], i_try
#                     )
#                 )
#                 if self.mode not in ["test"] and i_try > self._num_retries // 2:
#                     # let's try another one
#                     index = random.randint(0, len(self._path_to_videos) - 1)
#                 continue

#             # Decode video. Meta info is used to perform selective decoding.
#             frames = decode(
#                 container=video_container,
#                 sampling_rate=sampling_rate,
#                 num_frames=self.cfg.NUM_FRAMES,
#                 clip_idx=temporal_sample_index,
#                 num_clips=self.cfg.NUM_ENSEMBLE_VIEWS,
#                 video_meta=self._video_meta[index],
#                 target_fps=self.cfg.TARGET_FPS,
#                 backend=self.cfg.DECODING_BACKEND,
#                 max_spatial_scale=min_scale,
#             )
#             video_container.close()
#             # If decoding failed (wrong format, video is too short, and etc),
#             # select another video.
#             if frames is None:
#                 warnings.warn(
#                     "Failed to decode video idx {} from {}; trial {}".format(
#                         index, self._path_to_videos[index], i_try
#                     )
#                 )
#                 if self.mode not in ["test"] and i_try > self._num_retries // 2:
#                     # let's try another one
#                     index = random.randint(0, len(self._path_to_videos) - 1)
#                 continue

#             label = self._labels[index]
            
# #             print("111111111111")

# #             print(type(label))
#             # Perform color normalization.
#             # frames = tensor_normalize(
#             #     frames, self.cfg.MEAN, self.cfg.STD
#             # )
#             # frames=np.array(frames)
#             # frames=TensorNormalize()(frames)
#             # frames = frames.transpose(3, 0, 1, 2)
             
                
#             # frames=Spatial_sampling(min_scale,max_scale,spatial_sample_index)(frames)
#             # Perform data augmentation.
#             # frames = spatial_sampling(
#             #     frames,
#             #     spatial_idx=spatial_sample_index,
#             #     min_scale=min_scale,
#             #     max_scale=max_scale,
#             #     crop_size=crop_size,
#             #     random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
#             #     inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
#             # )

#             # if not self.cfg.MODEL.ARCH in ['vit']:
#             #     frames = pack_pathway_output(self.cfg, frames)
#             # else:
#             # Perform temporal sampling from the fast pathway.
#             # frames = [torch.index_select(
#             #     x,
#             #     1,
#             #     torch.linspace(
#             #         0, x.shape[1] - 1, self.cfg.DATA.NUM_FRAMES
#             #     ).long(),
#             # ) for x in frames]

#             return frames, np.array(label)
#         else:
#             raise RuntimeError(
#                 "Failed to fetch video after {} retries.".format(
#                     self._num_retries
#                 )
#             )

#     def __len__(self):
#         """
#         Returns:
#             (int): the number of videos in the dataset.
#         """
#         return len(self._path_to_videos)

# def create_dataset(args,device_num,local_rank, mode,shuffle=True):
#     cores = multiprocessing.cpu_count()
#     num_parallel_workers = min(min(64 * 1 * 2, int(cores / device_num)), 8)
#     # print("22222222222222222222222")
#     # print(num_parallel_workers)
#     # num_parallel_workers = 1
#     sampler = DistributedSampler(device_num,local_rank, shuffle=shuffle)
#     # sampler = mindspore.dataset.RandomSampler()   
    
#     if mode == "train":
#         min_scale = args.TRAIN_JITTER_SCALES_0
#         max_scale = args.TRAIN_JITTER_SCALES_1
#         mean=args.MEAN
#         std=args.STD
#         # if args.MULTIGRID_DEFAULT_S > 0:
#         #     min_scale = int(round(float(min_scale)* crop_size / self.cfg.MULTIGRID.DEFAULT_S))
#         video_dataset = UCF101(cfg=args, mode="train", num_retries=10)
#         column_names=["frames","label"]
#         data = GeneratorDataset(
#         source=video_dataset,
#         column_names=column_names,
#         num_parallel_workers=num_parallel_workers,
#         python_multiprocessing=False,
#         sampler=sampler,
#         )
#         type_cast_op = C2.TypeCast(mstype.int32)
#         trans=mindspore.dataset.transforms.Compose([TensorNormalize(),Spatial_sampling(min_scale,max_scale,spatial_idx=-1),C2.TypeCast(mstype.float32)])
#         # trans=mindspore.dataset.transforms.Compose([C2.TypeCast(mstype.float32)])
#         data = data.map(operations=trans, input_columns=["frames"], num_parallel_workers=num_parallel_workers)
#         data = data.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    
#     if mode == "val":
#         min_scale, max_scale, crop_size = (
#             [args.TEST_CROP_SIZE] * 3 if args.NUM_SPATIAL_CROPS > 1
#             else [args.TRAIN_JITTER_SCALES_0] * 2 + [args.TEST_CROP_SIZE]
#         )
#         assert len({min_scale, max_scale}) == 1
#         video_dataset = UCF101(cfg=args, mode="val", num_retries=10)
#         column_names=["frames", "label"]
#         data = GeneratorDataset(
#         source=video_dataset,
#         column_names=column_names,
#         num_parallel_workers=num_parallel_workers,
#         python_multiprocessing=False,
#         sampler=sampler,
#         )
        
#         trans=mindspore.dataset.transforms.Compose([TensorNormalize(),Spatial_sampling(min_scale,max_scale,spatial_idx=-1),C2.TypeCast(mstype.float32)])
#         type_cast_op = C2.TypeCast(mstype.int32)
#         # trans=mindspore.dataset.transforms.Compose([C2.TypeCast(mstype.float32)])
#         data = data.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
#         data = data.map(operations=trans, input_columns=["frames"], num_parallel_workers=num_parallel_workers)
        
#     data = data.batch(batch_size=2,drop_remainder=False)
#     data = data.repeat(1)
#     return data


    
    
    
    
    
    
# if __name__ == '__main__':

#     from utils.parser import parse_args, load_config
#     from tqdm import tqdm

#     args = parse_args()
#     args.cfg_file = "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
#     config = load_config(args)
#     config.DATA.PATH_TO_DATA_DIR = "/home/kanchanaranasinghe/repo/mmaction2/data/ucf101/splits"
#     config.DATA.PATH_PREFIX = "/home/kanchanaranasinghe/repo/mmaction2/data/ucf101/videos"
#     dataset = UCF101(cfg=config, mode="train", num_retries=10)
#     dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
#     print(f"Loaded train dataset of length: {len(dataset)}")
#     for idx, i in enumerate(dataloader):
#         print(idx, i[0].shape, i[1:])
#         if idx > 2:
#             break

#     test_dataset = UCF101(cfg=config, mode="val", num_retries=10)
#     test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4)
#     print(f"Loaded test dataset of length: {len(test_dataset)}")
#     for idx, i in enumerate(test_dataloader):
#         print(idx, i[0].shape, i[1:])
#         if idx > 2:
#             break
