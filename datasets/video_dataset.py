import os
import random
import warnings
from datasets.decoder import decode
from datasets.video_cv2 import get_video_container
import mindspore.dataset.transforms as C2
import mindspore
import mindspore.dataset
import numpy as np
from datasets.decoder import decode
from datasets.video_cv2 import get_video_container
from mindspore.dataset.transforms import TypeCast
import multiprocessing
from mindspore.dataset import DistributedSampler, GeneratorDataset
import mindspore.common.dtype as mstype
import multiprocessing
from datasets.transformers import GlobalTransform,GlobalTransform_ms,LocalTransform_ms,TransposeReshape_ms, LocalTransform, Change2, Change,Spatial_sampling, TensorNormalize,get_random_sampling_rate


class Kinetics:
    """
    UCF101 video loader. Construct the UCF101 video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, logger, num_retries=10):
        """
        Construct the UCF101 video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train mode, the data loader will take data from the
                train set, and sample one clip per video. For the val and
                test mode, the data loader will take data from relevent set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["pretrain", "finetune", "val", "test"], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.logger = self.cfg.logger

        self._video_meta = {}
        self._num_retries = num_retries
        self._split_idx = mode
        # For training mode, one single clip is sampled from every video. For validation or testing, NUM_ENSEMBLE_VIEWS
        # clips are sampled from every video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        if self.mode in ["pretrain", "finetune", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    self.cfg.NUM_ENSEMBLE_VIEWS * self.cfg.NUM_SPATIAL_CROPS
            )

        self.logger.info("Constructing Videos {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "test",
        }
        path_to_file = os.path.join(
            self.cfg.PATH_TO_DATA_DIR, "{}.csv".format(csv_file_name[self.mode])
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                        len(path_label.split(self.cfg.PATH_LABEL_SEPARATOR))
                        == 2
                )
                path, label = path_label.split(
                    self.cfg.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):  # _num_clips=30
                    self._path_to_videos.append(
                        os.path.join(self.cfg.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (len(self._path_to_videos) > 0), f"Failed to load Kinetics split {self._split_idx} from {path_to_file}"
        self.logger.info("Constructing Kinetics dataloader (size: {}) from {}".format(
            len(self._path_to_videos), path_to_file
        ))

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["pretrain", "finetune", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.TRAIN_JITTER_SCALES_0
            max_scale = self.cfg.TRAIN_JITTER_SCALES_1
            crop_size = self.cfg.TRAIN_CROP_SIZE
            if short_cycle_idx == 0:
                crop_size = int(
                    round(
                        self.cfg.SHORT_CYCLE_FACTORS_0
                        * self.cfg.DEFAULT_S
                    )
                )
            if short_cycle_idx == 1:
                crop_size = int(
                    round(
                        self.cfg.SHORT_CYCLE_FACTORS_1
                        * self.cfg.DEFAULT_S
                    )
                )
            if self.cfg.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (self._spatial_temporal_idx[index] // self.cfg.NUM_SPATIAL_CROPS)
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.NUM_SPATIAL_CROPS)
                if self.cfg.NUM_SPATIAL_CROPS > 1 else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.TEST_CROP_SIZE] * 3 if self.cfg.NUM_SPATIAL_CROPS > 1
                else [self.cfg.TRAIN_JITTER_SCALES_0] * 2 + [self.cfg.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = get_random_sampling_rate(
            self.cfg.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatedly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = get_video_container(
                    self._path_to_videos[index],
                    self.cfg.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DECODING_BACKEND,
                )
            except Exception as e:
                print(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
                self.logger.info("Failed to load video from {} with error {}".format(self._path_to_videos[index], e))
            # Select a random video if the current video was not able to access.
            total_frames = video_container.get(7)
            if video_container is None or (total_frames < 10):
                video_container.release()
                warnings.warn(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decode(
                container=video_container,
                sampling_rate=sampling_rate,
                num_frames=8,
                clip_idx=temporal_sample_index,
                num_clips=self.cfg.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.TARGET_FPS,
                backend=self.cfg.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                temporal_aug=self.mode == "pretrain" and not self.cfg.NO_RGB_AUG,
                two_token=self.cfg.TWO_TOKEN,
                rand_fr=self.cfg.RAND_FR
            )
            video_container.release()
            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                warnings.warn(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            if self.mode == "pretrain":
                frames = [x.transpose(0, 3, 1, 2) for x in frames]
                # ids_keep=self.generate_mask_matrix(4,0.25)
                ids_keep=self.random_masking(8,8,196)
                return frames[:2], frames[2:], ids_keep
            if self.mode in ["finetune", "val"]:
                return frames[0], frames[1], np.array(label)
                # return frames, np.array(label)
            elif self.mode == "test":
                return frames[0], frames[1], np.array(label), self._path_to_videos[index].split("/")[-1].rstrip(".mp4"), \
                       self._spatial_temporal_idx[index]
                # return frames, np.array(label), self._path_to_videos[index].split("/")[-1].rstrip(".mp4"), self._spatial_temporal_idx[index]
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )


    # def generate_mask_matrix(self, block_size, mask_rate):
    #     N, T, L, C = 8,8,196,768
    #     size = L ** 0.5
    #     size = int(size)
    #     mask = np.zeros((N, T, size, size))
    #     num_blocks = size // block_size
    #     num_mask_blocks = int(num_blocks * num_blocks * mask_rate)
    #     for b in range(N):
    #         for f in range(T):
    #             mask_block_indices = np.random.choice(num_blocks * num_blocks, num_mask_blocks, replace=False)
    #             for index in mask_block_indices:
    #                 row = index // num_blocks
    #                 col = index % num_blocks
    #                 mask[b, f, row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size] = 1 
    #     mask = mask.reshape(N,T, -1)   
    #     keep_len=mask[0,0].sum().astype(np.int32)
    #     ids_shuffle = np.argsort(mask, axis=-1).astype(np.int32)  
    #     ids_keep=ids_shuffle[:,:,-keep_len-4:]
    #     return ids_keep  

    def random_masking(self,N,T,L):
        len_keep = int(L * 0.25)
        noise = np.random.rand(N,T,L) 
        ids_shuffle = np.argsort(noise, axis=-1).astype(np.int32)
        ids_keep = ids_shuffle[:,:,:len_keep]
        return ids_keep  


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)


def create_dataset(args, mode='pretrain', shuffle=True):
    cores = multiprocessing.cpu_count()
    num_parallel_workers = max(min(args.batch_size * args.repeat_aug * 2, int(cores / args.device_num)), 8)
    sampler = DistributedSampler(args.device_num, args.local_rank, shuffle=shuffle)

    if mode == "pretrain":
        min_scale = args.TRAIN_JITTER_SCALES_0
        max_scale = args.TRAIN_JITTER_SCALES_1
        video_dataset = Kinetics(cfg=args, mode="pretrain", logger=args.logger, num_retries=10)
        column_names = ["g_frames", "l_frames", "ids_keep"]
        data = GeneratorDataset(
            source=video_dataset,
            column_names=column_names,
            num_parallel_workers=8,
            python_multiprocessing=False,
            sampler=sampler,
        )
        # g_t = mindspore.dataset.transforms.Compose([
        #     GlobalTransform(),
        #     Change(),
        #     TypeCast(mstype.float32)
        # ])
        # l_t = mindspore.dataset.transforms.Compose([
        #     LocalTransform(),
        #     Change(),
        #     TypeCast(mstype.float32)
        # ])

        # data = data.map(operations=g_t, input_columns=["g_frames"], num_parallel_workers=num_parallel_workers)
        # data = data.map(operations=l_t, input_columns=["l_frames"], num_parallel_workers=num_parallel_workers)

        g_t = mindspore.dataset.transforms.Compose([
            TransposeReshape_ms(),
            GlobalTransform_ms(),
            Change(),
            TypeCast(mstype.float32) 
        ])

        l_t = mindspore.dataset.transforms.Compose([
            TransposeReshape_ms(),
            LocalTransform_ms(),
            Change(),
            TypeCast(mstype.float32)
        ])

        data = data.map(operations=g_t, input_columns=["g_frames"], num_parallel_workers=num_parallel_workers,python_multiprocessing=True,)
        data = data.map(operations=l_t, input_columns=["l_frames"], num_parallel_workers=num_parallel_workers,python_multiprocessing=True,)


    elif mode in ["finetune", "val"]:
        min_scale = args.TRAIN_JITTER_SCALES_0
        max_scale = args.TRAIN_JITTER_SCALES_1
        video_dataset = Kinetics(cfg=args, mode=mode, logger=args.logger, num_retries=10)
        column_names = ["slow_frames", "fast_frames", "label"]
        # column_names=["slow_frames","label"]
        data = GeneratorDataset(
            source=video_dataset,
            column_names=column_names,
            num_parallel_workers=8,
            python_multiprocessing=False,
            sampler=sampler,
        )
        type_cast_op = C2.TypeCast(mstype.int32)
        slow_trans = mindspore.dataset.transforms.Compose(
            [TensorNormalize(), Spatial_sampling(min_scale, max_scale, spatial_idx=-1), C2.TypeCast(mstype.float32)])
        fast_trans = mindspore.dataset.transforms.Compose(
            [TensorNormalize(), Spatial_sampling(min_scale, max_scale, spatial_idx=-1, crop_size=96),
             C2.TypeCast(mstype.float32)])
        data = data.map(operations=slow_trans, input_columns=["slow_frames"], num_parallel_workers=num_parallel_workers)
        data = data.map(operations=fast_trans, input_columns=["fast_frames"], num_parallel_workers=num_parallel_workers)
        data = data.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)

    #     elif mode == "val":
    #         min_scale = args.TRAIN_JITTER_SCALES_0
    #         max_scale = args.TRAIN_JITTER_SCALES_1
    #         video_dataset = Kinetics(cfg=args, mode="val", logger=args.logger, num_retries=10)
    #         column_names=["slow_frames","fast_frames","label"]
    #         data = GeneratorDataset(
    #         source=video_dataset,
    #         column_names=column_names,
    #         num_parallel_workers=1,
    #         python_multiprocessing=False,
    #         sampler=sampler,
    #         )

    #         slow_trans=mindspore.dataset.transforms.Compose([TensorNormalize(),Spatial_sampling(min_scale,max_scale,spatial_idx=-1),C2.TypeCast(mstype.float32)])
    #         fast_trans=mindspore.dataset.transforms.Compose([TensorNormalize(),Spatial_sampling(min_scale,max_scale,spatial_idx=-1,crop_size=96),C2.TypeCast(mstype.float32)])
    #         type_cast_op = C2.TypeCast(mstype.int32)
    #         data = data.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    #         data = data.map(operations=slow_trans, input_columns=["slow_frames"], num_parallel_workers=num_parallel_workers)
    #         data = data.map(operations=fast_trans, input_columns=["fast_frames"], num_parallel_workers=num_parallel_workers)

    elif mode == "test":
        min_scale, max_scale, crop_size = (
            [args.TEST_CROP_SIZE] * 3 if args.NUM_SPATIAL_CROPS > 1
            else [args.TRAIN_JITTER_SCALES_0] * 2 + [args.TEST_CROP_SIZE]
        )
        assert len({min_scale, max_scale}) == 1
        video_dataset = Kinetics(cfg=args, mode="test", logger=args.logger, num_retries=10)
        column_names = ["slow_frames", "fast_frames", "label", "sample", "index"]
        # column_names=["slow_frames", "label","sample","index"]
        data = GeneratorDataset(
            source=video_dataset,
            column_names=column_names,
            num_parallel_workers=8,
            python_multiprocessing=False,
            sampler=sampler,
        )

        trans = mindspore.dataset.transforms.Compose(
            [TensorNormalize(), Spatial_sampling(min_scale, max_scale, spatial_idx=-1), C2.TypeCast(mstype.float32)])
        type_cast_op = C2.TypeCast(mstype.int32)
        slow_trans = mindspore.dataset.transforms.Compose(
            [TensorNormalize(), Spatial_sampling(min_scale, max_scale, spatial_idx=1), C2.TypeCast(mstype.float32)])
        fast_trans = mindspore.dataset.transforms.Compose(
            [TensorNormalize(), Spatial_sampling(min_scale, max_scale, spatial_idx=1, crop_size=96),
             C2.TypeCast(mstype.float32)])
        data = data.map(operations=slow_trans, input_columns=["slow_frames"], num_parallel_workers=num_parallel_workers)
        data = data.map(operations=fast_trans, input_columns=["fast_frames"], num_parallel_workers=num_parallel_workers)
        data = data.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)

    data = data.batch(batch_size=args.batch_size, drop_remainder=True)
    return data
