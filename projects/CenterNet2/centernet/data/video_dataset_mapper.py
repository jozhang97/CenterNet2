# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.transforms.augmentation import AugmentationList
from detectron2.structures import Boxes, BoxMode, Instances

from .custom_build_augmentation import build_custom_augmentation
from .custom_dataset_mapper import custom_annotations_to_instances
from .custom_dataset_mapper import custom_transform_instance_annotations
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop

__all__ = ["VideoDatasetMapper"]

class VideoDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        min_video_len: int = 1,
        max_video_len: int = 16,
    ):
        self.is_train = is_train
        self.augmentations = augmentations
        self.image_format = image_format
        self.min_video_len = min(min_video_len, max_video_len)
        self.max_video_len = max_video_len

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # ret = super().from_config(cfg, is_train)
        ret = {}
        ret['min_video_len'] = cfg.INPUT.VID.MIN_VIDEO_LEN
        ret['max_video_len'] = cfg.INPUT.VID.MAX_VIDEO_LEN
        if cfg.INPUT.CUSTOM_AUG != '':
            ret['augmentations'] = build_custom_augmentation(cfg, is_train)
        else:
            ret['augmentations'] = []
        return ret

    def __call__(self, video_dict):
        '''
        :params video_dict: {'video_id': int, 'images': [{'image_id', 'annotations': []}]}
        :return: List[Dict(images, instances)]
        '''
        ## Pick frames
        n_images = len(video_dict['images'])
        if self.is_train:
            n_frames = np.random.randint(self.min_video_len, self.max_video_len + 1)
            n_frames = min(n_images, n_frames)
        else:
            n_frames = n_images
        st = np.random.randint(n_images - n_frames + 1)
        images_dict = copy.deepcopy(video_dict['images'][st: st + n_frames])

        ## Load and apply transforms to frames and annotations
        ret = []
        for i, dataset_dict in enumerate(images_dict):
            fn = dataset_dict['file_name']
            # TODO try jpg and jpeg
            image = utils.read_image(fn, format=self.image_format)
            aug_input = T.StandardAugInput(image)
            transforms = aug_input.apply_augmentations(self.augmentations)
            image = aug_input.image
            image_shape = image.shape[:2]  # h, w

            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if 'annotations' in dataset_dict:
                annos = dataset_dict['annotations']
                for a in annos:
                    bbox = BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS)
                    a['bbox'] = transforms.apply_box(np.array([bbox]))[0]
                    a['bbox_mode'] = BoxMode.XYXY_ABS
                    # TODO handle crowd, clip box
                instances = custom_annotations_to_instances(annos, image_shape, with_inst_id=True)
                dataset_dict['instances'] = utils.filter_empty_instances(instances)
            ret.append(dataset_dict)
        return ret


'''
Deprecated
- this is XYZ's version, I use the barebones version above
'''

class XYZVideoDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        min_video_len: int = 1,
        max_video_len: int = 16,
        not_clip_box: bool = False,
        sample_range: float = 1.,
        dynamic_scale: bool = False,
        gen_image_motion: bool = False,
        gen_copy_paste: bool = False,
    ):
        """
        add instance_id # with_crowd, with_distill_score keys
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = augmentations
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk = precomputed_proposal_topk
        self.recompute_boxes = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))
        self.min_video_len = min(min_video_len, max_video_len)
        self.max_video_len = max_video_len
        self.not_clip_box = not_clip_box
        self.sample_range = sample_range
        self.dynamic_scale = dynamic_scale
        self.gen_image_motion = gen_image_motion
        self.gen_copy_paste = gen_copy_paste
        if self.gen_image_motion and is_train:
            self.motion_augmentations = [
                EfficientDetResizeCrop(
                    augmentations[0].target_size[0], (0.8, 1.2))]

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        # ret['with_inst_id'] = cfg.MODEL.REID_ON
        ret['min_video_len'] = cfg.INPUT.VID.MIN_VIDEO_LEN
        ret['max_video_len'] = cfg.INPUT.VID.MAX_VIDEO_LEN
        ret['not_clip_box'] = cfg.INPUT.NOT_CLIP_BOX
        ret['sample_range'] = cfg.INPUT.VID.SAMPLE_RANGE
        ret['dynamic_scale'] = cfg.INPUT.VID.DYNAMIC_SCALE
        ret['gen_image_motion'] = cfg.INPUT.VID.GEN_IMAGE_MOTION
        ret['gen_copy_paste'] = cfg.INPUT.VID.GEN_COPY_PASTE
        return ret

    def __call__(self, video_dict):
        """
        video_dict: {'video_id': int, 'images': [{'image_id', 'annotations': []}]}
        """
        if self.is_train:
            num_frames = np.random.randint(
                self.max_video_len - self.min_video_len + 1) + self.min_video_len
            num_frames = min(len(video_dict['images']), num_frames)
        else:
            num_frames = len(video_dict['images'])
        st = np.random.randint(len(video_dict['images']) - num_frames + 1)
        gen_image_motion = self.gen_image_motion and self.is_train and \
                           len(video_dict['images']) == 1

        if self.dynamic_scale and self.is_train and not gen_image_motion:
            image = utils.read_image(
                video_dict['images'][st]["file_name"], format=self.image_format)
            aug_input = T.StandardAugInput(image)
            transforms = aug_input.apply_augmentations(self.augmentations)
            auged_size = max(transforms[0].scaled_w, transforms[0].scaled_h)
            target_size = max(transforms[0].target_size)
            max_frames = int(num_frames * (target_size / auged_size) ** 2)
            if max_frames > self.max_video_len:
                num_frames = np.random.randint(
                    max_frames - self.min_video_len + 1) + self.min_video_len
                num_frames = min(self.max_video_len * 2, num_frames)
                num_frames = min(len(video_dict['images']), num_frames)
            # print('num_frames', num_frames)
        else:
            transforms = None

        if gen_image_motion:
            num_frames = num_frames = np.random.randint(
                self.max_video_len - self.min_video_len + 1) + self.min_video_len
            images_dict = [copy.deepcopy(
                video_dict['images'][0]) for _ in range(num_frames)]
            image = utils.read_image(
                video_dict['images'][0]["file_name"], format=self.image_format)
            width, height = image.shape[1], image.shape[0]
            aug_input = T.StandardAugInput(image)
            transforms_st = aug_input.apply_augmentations(self.motion_augmentations)
            transforms_ed = aug_input.apply_augmentations(self.motion_augmentations)
            transforms_list = []
            for x in range(num_frames):
                trans = copy.deepcopy(transforms_st)
                # trans[0].scaled_h += (transforms_ed[0].scaled_h - \
                #     transforms_st[0].scaled_h) * x // (num_frames - 1)
                # trans[0].scaled_w += (transforms_ed[0].scaled_w - \
                #     transforms_st[0].scaled_w) * x // (num_frames - 1)
                trans[0].offset_x += (transforms_ed[0].offset_x - \
                                      transforms_st[0].offset_x) * x // (num_frames - 1)
                trans[0].offset_y += (transforms_ed[0].offset_y - \
                                      transforms_st[0].offset_y) * x // (num_frames - 1)
                trans[0].img_scale += (transforms_ed[0].img_scale - \
                                       transforms_st[0].img_scale) * x / (num_frames - 1)
                trans[0].scaled_h = int(height * trans[0].img_scale)
                trans[0].scaled_w = int(width * trans[0].img_scale)
                transforms_list.append(trans)
            # import pdb; pdb.set_trace()
        elif self.sample_range > 1. and self.is_train:
            ed = min(st + int(self.sample_range * num_frames), len(video_dict['images']))
            num_frames = min(num_frames, ed - st)
            inds = sorted(
                np.random.choice(range(st, ed), size=num_frames, replace=False))
            images_dict = copy.deepcopy([video_dict['images'][x] for x in inds])
        else:
            images_dict = copy.deepcopy(video_dict['images'][st: st + num_frames])

        ret = []
        for i, dataset_dict in enumerate(images_dict):
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.StandardAugInput(image)
            if gen_image_motion:
                transforms = transforms_list[i]
                image = transforms.apply_image(image)
            elif transforms is None:
                transforms = aug_input.apply_augmentations(self.augmentations)
                image = aug_input.image
            else:
                image = transforms.apply_image(image)

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                all_annos = [
                    (custom_transform_instance_annotations(
                        obj, transforms, image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                        not_clip_box=self.not_clip_box,
                    ), obj.get("iscrowd", 0))
                    for obj in dataset_dict.pop("annotations")
                ]
                annos = [ann[0] for ann in all_annos if ann[1] == 0]
                instances = custom_annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format,
                    with_inst_id=True
                )

                del all_annos
                dataset_dict["instances"] = utils.filter_empty_instances(instances)
            ret.append(dataset_dict)
        return ret

