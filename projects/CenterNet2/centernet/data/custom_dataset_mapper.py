# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.structures import Keypoints, PolygonMasks, BitMasks

__all__ = ["CustomDatasetMapper"]


def custom_annotations_to_instances(annos, image_size, mask_format="polygon", \
                                    with_inst_id=False, with_track_id=False):
    """
    Add instance id
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            # TODO check type and provide better error
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    if with_inst_id:
        instance_ids = [obj.get('instance_id', 0) for obj in annos]
        target.gt_instance_ids = torch.tensor(instance_ids, dtype=torch.int64)

    if with_track_id:
        tids = [obj['track_id'] for obj in annos]
        target.gt_track_ids = torch.tensor(tids, dtype=torch.int64)

    return target


def custom_transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None,
    not_clip_box=False,
):
    """
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    if not not_clip_box:
        bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
        annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    else:
        bbox = transforms.apply_box(np.array([bbox]))[0]
        annotation["bbox"] = bbox

    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


class CustomDatasetMapper(DatasetMapper):
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
        # with_crowd: bool = False,
        # with_distill_score: bool = False,
        with_inst_id: bool = False,
        not_clip_box: bool = False,
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

        # self.with_crowd = with_crowd
        # self.with_distill_score = with_distill_score
        self.with_inst_id = with_inst_id
        self.not_clip_box = not_clip_box

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        # ret['with_crowd'] = cfg.WITH_CROWD
        # ret['with_distill_score'] = cfg.WITH_DISTILL_SCORE
        ret['with_inst_id'] = cfg.INPUT.WITH_INST_ID
        ret['not_clip_box'] = cfg.INPUT.NOT_CLIP_BOX
        return ret

    def __call__(self, dataset_dict):
        """
        include is_crowd and instance_id
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

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
                    not_clip_box=self.not_clip_box
                ), obj.get("iscrowd", 0))
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]

            if self.with_inst_id:
                instances = custom_annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format,
                    with_inst_id=True
                )
            else:
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )

            # if self.with_distill_score:
            #     instances.gt_distill_scores = torch.tensor(
            #         [ann['score'] if 'score' in ann else 1. for ann in annos],
            #         dtype=torch.float32,
            #     )

            # if self.with_crowd:
            #     crowds = [ann[0]['bbox'] for ann in all_annos if ann[1] != 0]
            #     instances._crowd_boxes = Boxes(crowds)
            del all_annos

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict