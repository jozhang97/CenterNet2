# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import operator
import torch
import torch.utils.data
import json
from detectron2.utils.comm import get_world_size

from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data import samplers
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.data.samplers import InferenceSampler
from detectron2.data.build import worker_init_reset_seed, print_instances_class_histogram
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.data.build import check_metadata_consistency
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm
import itertools
import math
from collections import defaultdict
from typing import Optional

from .custom_dataset_dataloader import MultiDatasetSampler
from .video_dataset_loader import get_video_dataset_dicts


def single_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    assert len(batch) == 1
    return batch[0]


def build_video_train_loader(cfg, mapper):
    """
    Modified from detectron2.data.build.build_custom_train_loader, but supports
    different samplers
    """
    dataset_dicts = get_video_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        gen_inst_id=True,
    )
    sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
    for d in dataset_dicts:
        sizes[d['dataset_source']] += 1
    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training videos {}".format(sampler_name))
    if len(cfg.DATASETS.TRAIN) > 1:
        assert sampler_name == 'MultiDatasetSampler'

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "MultiDatasetSampler":
        sampler = MultiDatasetSampler(cfg, sizes, dataset_dicts)
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    world_size = get_world_size()
    batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size
    assert batch_size == 1
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=single_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )


def build_video_test_loader(cfg, dataset_name, mapper):
    """
    """
    assert comm.is_main_process()
    dataset = get_video_dataset_dicts(
        [dataset_name],
        filter_empty=False,
    )
    dataset = DatasetFromList(dataset, copy=False)
    dataset = MapDataset(dataset, mapper)

    sampler = SingleGPUInferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=batch_sampler,
        collate_fn=single_batch_collator,
    )
    return data_loader


def split_video_dataset(dataset, test_len, stride):
    assert stride == test_len
    ret = []
    print('Spliting {} videos into shorter clips'.format(len(dataset)))
    for video in dataset:
        video_len = len(video['images'])
        st, ed = 0, 0
        num_clips = (video_len - 1) // test_len + 1
        for _ in range(num_clips):
            ed = min(st + test_len, video_len)
            clip = {
                'video_id': video['video_id'],
                'dataset_source': video['dataset_source'],
                'images': copy.deepcopy(video['images'][st: ed]),
            }
            ret.append(clip)
            st += stride
    print('#clips', len(ret))
    return ret



class SingleGPUInferenceSampler(Sampler):
    def __init__(self, size: int):
        """
        self._world_size = 1
        self._rank = 0
        """
        self._size = size
        assert size > 0
        # self._rank = comm.get_rank()
        self._rank = 0
        # self._world_size = comm.get_world_size()
        self._world_size = 1

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)