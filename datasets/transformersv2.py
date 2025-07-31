import numpy as np
from scipy.ndimage import zoom
import math
import random
from mindspore import dtype as mstype
import mindspore.dataset.transforms 


def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate

def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


class TransposeReshape:
    def __init__(self, repeat_aug=1):
        self._repeat_aug = repeat_aug
    
    def __call__(self, frames):
        C,T,H,W = frames.shape
        frames = frames.transpose(2, 3, 0, 1) # h, w, r, c, t
        frames = frames.reshape(H, W,C*T)
        return frames

class TransposeReshape_ms_g:
    def __init__(self, repeat_aug=1):
        self._repeat_aug = repeat_aug
    
    def __call__(self, frames_g0):
        T0,C,H,W = frames_g0.shape
        frames_g0 = frames_g0.transpose(2, 3, 0, 1)   #h,w,t,c
        frames_g0 = frames_g0.reshape(H, W,T0*C)
        
        return frames_g0
    
class TransposeReshape_ms_l:
    def __init__(self, repeat_aug=1):
        self._repeat_aug = repeat_aug
    
    def __call__(self, frames_l0):
        T0,C,H,W = frames_l0[0].shape
        frames_l0 = [x.transpose(2, 3, 0, 1) for x in frames_l0]   #h,w,t,c
        frames_l0 = [x.reshape(H, W,T0*C) for x in frames_l0]
        
        #         T1,C,H,W = frames_l1[0].shape
        #         frames_l1 = [x.transpose(2, 3, 0, 1) for x in frames_l1]   #h,w,t,c
        #         frames_l1 = [x.reshape(H, W,T1*C) for x in frames_l1]

        #         T2,C,H,W = frames_l2[0].shape
        #         frames_l2 = [x.transpose(2, 3, 0, 1) for x in frames_l2]   #h,w,t,c
        #         frames_l2 = [x.reshape(H, W,T2*C) for x in frames_l2]

        #         T3,C,H,W = frames_l3[0].shape
        #         frames_l3 = [x.transpose(2, 3, 0, 1) for x in frames_l3]   #h,w,t,c
        #         frames_l3 = [x.reshape(H, W,T3*C) for x in frames_l3]
        return frames_l0
    



class ReshapeTranspose_ms:
    def __init__(self,num_frames=8):
        # self._repeat_aug = repeat_aug
        self._num_frames = num_frames
        self.hwc2chw=vision.HWC2CHW()
    
    def __call__(self, frames,frames_nums):
        # self._num_frames=frames_nums
        frames=self.hwc2chw(frames)
        c, h, w = frames.shape
        frames = frames.reshape(frames_nums,3 ,h, w) # r, c, t, h, w

        return frames

    
    
class ReshapeTranspose:
    def __init__(self,num_frames=8):
        # self._repeat_aug = repeat_aug
        self._num_frames = num_frames
    
    def __call__(self, frames):
        c, h, w = frames.shape
        frames = frames.reshape(3, self._num_frames, h, w) # r, c, t, h, w
        frames=frames.astype(np.float32)
        # frames = frames.transpose(1, 0, 2, 3, 4) # r, c, t, h, w
        return frames




#v
class Random_short_side_scale_jitter:
    def __init__(self, min_size, max_size, boxes=None, inverse_uniform_sampling=False):
        self.min_size = min_size
        self.max_size = max_size
        self.boxes = boxes
        self.inverse_uniform_sampling = inverse_uniform_sampling

    def __call__(self, images):

        if self.inverse_uniform_sampling:
            size = int(
                round(1.0 / np.random.uniform(1.0 / self.max_size, 1.0 / self.min_size))
            )
        else:
            size = int(round(np.random.uniform(self.min_size, self.max_size)))#256

        height = images.shape[2]
        width = images.shape[3]
        if (width <= height and width == size) or (
                height <= width and height == size
        ):
            return images
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
            # if self.boxes is not None:
            #     self.boxes = self.boxes * float(new_height) / height
        else:
            new_width = int(math.floor((float(width) / height) * size))
            # if self.boxes is not None:
            #     self.boxes = self.boxes * float(new_width) / width
        # images=zoom(images, (1, 1, new_height / images.shape[2], new_width / images.shape[3]), order=1)
        images=images.transpose(0,2,3,1)
        images=mindspore.dataset.vision.Resize((new_height,new_width), interpolation=Inter.BILINEAR)(images)
        images=images.transpose(0,3,1,2)
        return images

#v
class Random_crop:
    def __init__(self, size, boxes=None):
        self.size = size
        self.boxes = boxes

    def __call__(self, images):
        if images.shape[2] == self.size and images.shape[3] == self.size:
            return images, None
        height = images.shape[2]
        width = images.shape[3]
        y_offset = 0
        if height > self.size:
            y_offset = int(np.random.randint(0, height - self.size))
        x_offset = 0
        if width > self.size:
            x_offset = int(np.random.randint(0, width - self.size))
        cropped = images[
                  :, :, y_offset: y_offset + self.size, x_offset: x_offset + self.size
                  ]

        # cropped_boxes = (
        #     crop_boxes(self.boxes, x_offset, y_offset) if self.boxes is not None else None
        # )

        # return cropped, cropped_boxes
        return cropped

#v
class Uniform_crop:
    def __init__(self, size, boxes=None):
        self.size = size
        self.boxes = boxes

    def __call__(self, images,spatial_idx):

        height = images.shape[2]
        width = images.shape[3]

        # print(spatial_idx)
        spatial_idx=spatial_idx%3
        # spatial_idx=np.random.randint(0,3)

        y_offset = int(math.ceil((height - self.size) / 2))
        x_offset = int(math.ceil((width - self.size) / 2))

        if height > width:
            if spatial_idx == 0:
                y_offset = 0
            elif spatial_idx == 2:
                y_offset = height - self.size
        else:
            if spatial_idx == 0:
                x_offset = 0
            elif spatial_idx == 2:
                x_offset = width - self.size
        cropped = images[
                  :, :, y_offset: y_offset + self.size, x_offset: x_offset + self.size
                  ]

        # cropped_boxes = (
        #     crop_boxes(self.boxes, x_offset, y_offset) if self.boxes is not None else None
        # )

        return cropped

#v
class Resize:
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, images):
        if isinstance(self.size, int):
            new_height, new_width = self.size, self.size
        else:
            new_height, new_width = self.size
        return zoom(images, (1, 1, new_height / images.shape[2], new_width / images.shape[3]), order=2)

#v
class RandomResizedCrop:
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.), interpolation='bilinear') -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.resize = Resize(size=self.size, mode=self.interpolation)

    def __call__(self, images):
        # images is global_1 or global_2
        height, width = images.shape[-2:]
        area = height * width
        non_central = False

        for _ in range(10):
            target_area = area * np.random.uniform(self.scale[0], self.scale[1])
            log_ratio = np.log(self.ratio)
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
                non_central = True

        if not non_central:
            # fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2
        y_offset, x_offset = i, j
        cropped = images[:, :, y_offset: y_offset + h, x_offset: x_offset + w]
        resized = self.resize(cropped)
        return np.array(resized)

#v
class Horizontal:
    def __init__(self, prob, boxes=None):
        self.prob = prob
        self.boxes = boxes

    def __call__(self, images):
        if self.boxes is None:
            flipped_boxes = None
        else:
            flipped_boxes = self.boxes.copy()

        if np.random.uniform() < self.prob:
            images = np.flip(images, axis=(-1))
            width = images.shape[3]
            if self.boxes is not None:
                flipped_boxes[:, [0, 2]] = width - self.boxes[:, [2, 0]] - 1
        return np.array(images)

#v
class Blend:
    def __call__(self, images1, images2, alpha):
        return images1 * alpha + images2 * (1 - alpha)

#v
class BrightnessJitter:
    def __init__(self, img_brightness):
        self.var = img_brightness
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        self.blend = Blend()

    def __call__(self, images):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        img_bright=np.zeros(images.shape)
        images = self.blend(images, img_bright,alpha)
        return np.array(images)

#v
class Grayscale:
    def __call__(self, images):
        img_gray = np.copy(images)
        gray_channel = (
                0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
        )
        img_gray[:, 0] = gray_channel
        img_gray[:, 1] = gray_channel
        img_gray[:, 2] = gray_channel
        return img_gray

#v
class ContrastJitter:
    def __init__(self, img_contrast):
        self.var = img_contrast
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        self.blend = Blend()
        self.grayscale = Grayscale()

    def __call__(self, images):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        img_gray = self.grayscale(images)
        img_gray[:] = np.mean(img_gray, axis=(1, 2, 3), keepdims=True)
        images = self.blend(images, img_gray, alpha)
        return images

#v
class SaturationJitter:
    def __init__(self, img_saturation):
        self.var = img_saturation
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        self.blend = Blend()
        self.grayscale = Grayscale()

    def __call__(self, images):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        img_gray = self.grayscale(images)
        images = self.blend(images, img_gray, alpha)
        return images

#v
class ColorJitter:
    def __init__(self, img_brightness=0.4, img_contrast=0.4, img_saturation=0.2):
        self.img_brightness = img_brightness
        self.img_contrast = img_contrast
        self.img_saturation = img_saturation
        self.brightness_jitter = BrightnessJitter(self.img_brightness)
        self.contrast_jitter = ContrastJitter(self.img_contrast)
        self.saturation_jitter = SaturationJitter(self.img_saturation)

    def __call__(self, images):
        jitter = []
        if self.img_brightness != 0:
            jitter.append("brightness")
        if self.img_contrast != 0:
            jitter.append("contrast")
        if self.img_saturation != 0:
            jitter.append("saturation")

        if len(jitter) > 0:
            order = np.random.permutation(np.arange(len(jitter)))
            for idx in range(0, len(jitter)):
                if jitter[order[idx]] == "brightness":
                    images = self.brightness_jitter(images)
                elif jitter[order[idx]] == "contrast":
                    images = self.contrast_jitter(images)
                elif jitter[order[idx]] == "saturation":
                    images = self.saturation_jitter(images)
        return images


#v
class FlipAndColorJitter:
    def __init__(self):
        self.horizontal = Horizontal(prob=0.5)
        self.color_jitter = ColorJitter()
        self.grayscale = Grayscale()

    def __call__(self, frames):
        frames= self.horizontal(images=frames)
        if np.random.uniform() < 0.8:
            frames = self.color_jitter(frames)
        if np.random.uniform() < 0.2:
            frames = self.grayscale(frames)
        return np.array(frames)


class FlipAndColorJitter_ms:
    def __init__(self):
        self.horizontal = vision.RandomHorizontalFlip(0.5)
        self.color_jitter = ColorJitter()
        # self.color_jitter = vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.2, hue=(0, 0))
        # self.grayscale = vision.Grayscale(num_output_channels=3)
        self.grayscale = Grayscale()
        self.reshapeTranspose_ms=ReshapeTranspose_ms()
        

    def __call__(self, frames,frames_num):
        frames= self.horizontal(frames)
        frames=self.reshapeTranspose_ms(frames,frames_num)
        if np.random.uniform() < 0.8:
            frames = self.color_jitter(frames)
        if np.random.uniform() < 0.2:
            frames = self.grayscale(frames)
        return frames


#v
class ColorNormalization:
    def __init__(self, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, images):
        assert len(self.mean) == images.shape[1], "channel mean not computed properly"
        assert (
                len(self.stddev) == images.shape[1]
        ), "channel stddev not computed properly"
        out_images = np.zeros_like(images)
        for idx in range(len(self.mean)):
            out_images[:, idx] = (images[:, idx] - self.mean[idx]) / self.stddev[idx]
        return out_images

class ColorNormalization_ms:
    def __init__(self, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, images):
        assert len(self.mean) == images.shape[0], "channel mean not computed properly"
        assert (
                len(self.stddev) == images.shape[1]
        ), "channel stddev not computed properly"
        out_images = np.zeros_like(images)
        for idx in range(len(self.mean)):
            out_images[idx] = (images[idx] - self.mean[idx]) / self.stddev[idx]
        return out_images


#v
class Normalize:
    def __init__(self):
        self.color_normalization = ColorNormalization()

    def __call__(self, frames):
        frames = self.color_normalization(frames)
        return frames


class GlobalTransform:           
    def __init__(self, global_crops_scale=(0.4, 1.0)):
        self.global_crops_scale = global_crops_scale
        self.random_resized_crop = RandomResizedCrop(size=224, scale=self.global_crops_scale, interpolation="bilinear")
        self.flip_and_color_jitter = FlipAndColorJitter()
        self.normalize = Normalize()
        self.rescale=vision.Rescale(rescale=1.0/255.0, shift=0)

    def __call__(self, images):
        result = []

        # global_transform1
        # images = [x.astype(float) / 255.0 for x in images]
        frames = images[0]
        frames=self.rescale(frames)
        frames = self.random_resized_crop(frames)
        frames = self.flip_and_color_jitter(frames)
        frames = self.normalize(frames)
        result.append(frames)

        # global_transform2
        frames = images[1]
        frames=self.rescale(frames)
        frames = self.random_resized_crop(frames)
        frames = self.flip_and_color_jitter(frames)
        frames = self.normalize(frames)
        result.append(frames)

        return np.array(result)


class GlobalTransform_ms:           
    def __init__(self, global_crops_scale=(0.4, 1.0)):
        self.global_crops_scale = global_crops_scale
        self.random_resized_crop = vision.RandomResizedCrop(size=224, scale=(0.4,1.0), ratio=(3. / 4., 4. / 3.),interpolation=Inter.BICUBIC)
        self.flip_and_color_jitter = FlipAndColorJitter_ms()
        self.normalize = Normalize()
        self.rescale=vision.Rescale(rescale=1.0/255.0, shift=0)

    def __call__(self, image0):

        image0=self.rescale(image0)
        image0 = self.random_resized_crop(image0)
        image0 = self.flip_and_color_jitter(image0,16)
        image0 = self.normalize(image0)

        return image0





class GlobalTransform2:
    def __init__(self, global_crops_scale=(0.25, 1.0)):
        self.global_crops_scale = global_crops_scale
        self.random_resized_crop = RandomResizedCrop(size=224, scale=self.global_crops_scale, interpolation="bilinear")
        self.flip_and_color_jitter = FlipAndColorJitter()
        self.normalize = Normalize()

    def __call__(self, frames):
        # global_transform1

        frames = self.random_resized_crop(frames)
        frames = self.flip_and_color_jitter(frames)
        frames = self.normalize(frames)

        return np.array(frames)

class LocalTransform_ms:
    def __init__(self, local_crops_scale=(0.05, 0.4)):
        # self.local_crops_scale = local_crops_scale
        self.random_resized_crop = vision.RandomResizedCrop(size=224, scale=(0.05, 0.4), ratio=(3. / 4., 4. / 3.),interpolation=Inter.BICUBIC)
        self.flip_and_color_jitter = FlipAndColorJitter_ms()
        self.normalize = Normalize()
        self.rescale=vision.Rescale(rescale=1.0/255.0, shift=0)

    def __call__(self, images):
        result = []
        for i,image in enumerate(images):
            frames=self.rescale(image)
            frames = self.random_resized_crop(frames)
            frames = self.flip_and_color_jitter(frames,8)
            frames = self.normalize(frames)
            result.append(frames)
        return result



class LocalTransform:
    def __init__(self, local_crops_scale=(0.05, 0.4)):
        self.local_crops_scale = local_crops_scale
        self.random_resized_crop = RandomResizedCrop(size=96, scale=self.local_crops_scale, interpolation="bilinear")
        self.normalize = Normalize()
        self.flip_and_color_jitter = FlipAndColorJitter()
        self.rescale=vision.Rescale(rescale=1.0/255.0, shift=0)

    def __call__(self, images):
        result = []
        # images = [x.astype(float) / 255.0 for x in images]
        for i in images:
            frames=self.rescale(i)
            frames = self.random_resized_crop(frames)
            frames = self.flip_and_color_jitter(frames)
            frames = self.normalize(frames)
            # frames=frames.transpose(1, 0, 2, 3)
            result.append(frames)
        return np.array(result)


class Change_g:
    def __call__(self, image0):

        image0=image0.transpose(1, 0, 2, 3)
        
        image0=image0.astype(np.float32)
        return image0

    
class Change_l:
    def __call__(self, images):
        
        image = [x.transpose(1, 0, 2, 3).astype(np.float32) for x in images]

        return image
    

class Change2:
    def __call__(self, frames):

        frames = frames.transpose(1, 0, 2, 3) # C，T，H，W
        return np.array(frames)

import mindspore
from mindspore.dataset import (DistributedSampler, SequentialSampler,
                               transforms, vision)
from mindspore.dataset.vision import Inter
class TestAUG:
    def __init__(self,
                rescale=1.0/255.0,
                shift=0,
                mean=(0.45, 0.45, 0.45),
                std=(0.225, 0.225, 0.225),
                # repeat_aug=1,
                short_side_size=224,
                interpolation=Inter.BILINEAR,
                cropsize=224,
                test_num_spatial_crops=3,
                num_frames=16):
        self._rescale = rescale
        self._shift = shift
        self._mean = mean
        self._std = std
        # self._repeat_aug = repeat_aug
        self._short_side_size = short_side_size
        self._interpolation = interpolation
        self._cropsize = cropsize
        self._test_num_spatial_crops = test_num_spatial_crops
        self._num_frames = num_frames

        # self.rescale = vision.Resize(rescale=self._rescale, shift=self._shift)
        # self.normalize = vision.Normalize(mean=self._mean, std=self._std)
        # self.TransposeReshape = TransposeReshape(self._repeat_aug)
        # self.resize = vision.Resize(size=self._short_side_size, interpolation=self._interpolation)
        self.aug1 = mindspore.dataset.transforms.Compose([vision.Rescale(rescale=self._rescale, shift=self._shift), 
                            TensorNormalize(mean=self._mean, std=self._std),
                            TransposeReshape(),
                            vision.Resize(size=self._short_side_size, interpolation=self._interpolation)])
        self.aug2 = mindspore.dataset.transforms.Compose([vision.HWC2CHW(),
                             ReshapeTranspose(num_frames=self._num_frames)])

        
    def __call__(self, frames, index):
        frames = self.aug1(frames)
        h, w, c = frames.shape
        test_num_spatial_crops = self._test_num_spatial_crops
        # 如果只测试单个视角，则和验证集增强相同，采用中心裁剪，所以test_num_spatial_crops=3
        if test_num_spatial_crops == 1:
            test_num_spatial_crops = 3
            spatial_sample_index = 1
        else:
            spatial_sample_index = index % test_num_spatial_crops if test_num_spatial_crops > 1 else 1
        spatial_step = 1.0 * (max(h, w) - self._short_side_size) / (test_num_spatial_crops - 1)
        spatial_start = int(spatial_sample_index * spatial_step)
        if h >= w:
            frames = frames[spatial_start:spatial_start + self._short_side_size, :, :]
        else:
            frames = frames[:, spatial_start:spatial_start + self._short_side_size, :]
        frames = self.aug2(frames)
        return frames, index





# v
class TensorNormalize:  
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """

    def __init__(self, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        if isinstance(mean, list) or isinstance(mean, tuple):
            mean = np.array(mean)
        if isinstance(std, list) or isinstance(std, tuple):
            std = np.array(std)
        self._mean = mean
        self._std = std
        self.rescale=vision.Rescale(rescale=1.0/255.0, shift=0)

    def __call__(self, tensor,index):
        # if tensor.dtype.name == "uint8":
        #     # tensor = np.ascontiguousarray(tensor)
        #     # tensor.dtype = "float32"
        #     tensor = tensor.astype(np.float32)
        #     tensor = tensor / 255.0
        # mean = np.array(self.mean)
        # std = np.array(self.std)
        tensor=self.rescale(tensor)
        tensor = tensor - self._mean
        tensor = tensor / self._std
        tensor = tensor.transpose(3, 0, 1, 2)   # T H W C -> C T H W.
        return np.array(tensor),index



class Spatial_sampling:
    def __init__(self, min_scale, max_scale,spatial_idx=-1 ,crop_size=224, random_horizontal_flip=True,
                 inverse_uniform_sampling=False):
        self.random_short_side_scale_jitter = Random_short_side_scale_jitter(min_scale, max_scale,
                                                                             inverse_uniform_sampling)
        self.Random_crop = Random_crop(crop_size)
        self.Horizontal = Horizontal(0.5)
        self.spatial_idx=spatial_idx
        # self.it=0
        self.Uniform_crop = Uniform_crop(crop_size)
        self.random_horizontal_flip = random_horizontal_flip
        self.typecast=mindspore.dataset.transforms.TypeCast(mstype.float32)

    def __call__(self, frames,index):
        assert self.spatial_idx in [-1, 0, 1, 2]
        if self.spatial_idx == -1:
            frames = self.random_short_side_scale_jitter(frames)
            frames = self.Random_crop(frames)
            if self.random_horizontal_flip:
                frames = self.Horizontal(frames)
            frames=frames.astype(np.float32)
            # frames=self.typecast(frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            # assert len({min_scale, max_scale, crop_size}) == 1
            frames= self.random_short_side_scale_jitter(frames)
            frames= self.Uniform_crop(frames,index)
            frames=frames.astype(np.float32)
            # frames=self.typecast(frames)
        # self.it=self.it+1

        return frames,index


# class Dim_Trans:
#     def __call__(self,g_images):
#         # g_images=g_images.transpose(1,0,2,3,4,5)
#         g_images=list(np.split(g_images,g_images.shape[0],axis=0))

#         # images=g_images+l_images
#         images=[]
#         for i, x in enumerate(g_images):
#                 images[i]=x[0,:,:,:,:,:]
#         return images

#vv
class ValChange:
    def __call__(self, frames):
        frames = [x.transpose(3, 0, 1, 2) for x in frames]
        return np.array(frames)
