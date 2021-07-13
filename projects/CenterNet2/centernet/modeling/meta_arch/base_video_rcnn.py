from typing import Dict, List, Optional, Tuple
from overrides import overrides

import torch

from detectron2.structures import Instances
from detectron2.config import configurable

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

@META_ARCH_REGISTRY.register()
class BaseVideoRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, conf_thresh=0.3, **kwargs):
        super().__init__(**kwargs)
        self.conf_thresh = conf_thresh

    @classmethod
    @overrides
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['conf_thresh'] = cfg.EVAL.CONF_THRESH
        return ret

    @overrides
    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        '''
        :params batched_inputs: video
        :return: List[Dict] list of detections with keys {instances}
            Each detection is a unique track
        '''
        predictions = super().inference(batched_inputs, detected_instances, do_postprocess)
        st_id = 0
        for pred in predictions:
            pred['instances'] = pred['instances'][pred['instances'].scores > self.conf_thresh]
            pred['instances'].pred_track_ids = torch.arange(st_id, st_id + len(pred['instances']))
            st_id += len(pred['instances'])
        return predictions