import itertools
import json
import os

from fvcore.common.file_io import PathManager

from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.structures import BoxMode

from .eval_tao import setup_cfg, eval_tao
from centernet.data.datasets.tao import ALL_TAO_SPLITS
from pathlib import Path

def instances_to_tao_json(instances, img_id):
    """
    Copied from coco_evaluation.py, adds track_id, removed segmentation, keypoints
    Dump an "Instances" object to a TAO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    tids = instances.pred_track_ids.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "track_id": tids[k],
        }
        results.append(result)
    return results


class MOTEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir, use_fast_impl=use_fast_impl)
        self.dataset_name = dataset_name
        self._output_dir = Path(output_dir)
        self.oracle = cfg.EVAL.ORACLE

    def process(self, inputs, outputs):
        '''
        Copied from coco_evaluation.py
        Identical to COCOEvaluator.process except instances_to_X_json
        :loads self._predictions: List[Dict] of {image_id, instances as coco}
        '''
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_tao_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        assert img_ids is None
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        file_path = self._output_dir / "coco_instances_results.json"
        self._logger.info(f"Saving results to {file_path}")
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        if 'tao' in self.dataset_name:
            # _output_dir: /path/to/trackers_folder/tracker_to_eval/inference_{split_name}
            split = str(self._output_dir.name).replace('inference_', '')
            json_file = Path(ALL_TAO_SPLITS[split][1]).name.replace('.json', '')
            eval_cfgs = setup_cfg({
                'GT_FOLDER': f'/scratch/cluster/jozhang/datasets/TAO/annotations/{json_file}',
                'TRACKERS_FOLDER': str(self._output_dir.parent.parent),
                'TRACKERS_TO_EVAL': [self._output_dir.parent.name],
                'TRACKER_SUB_FOLDER': self._output_dir.name,
                'SPLIT_TO_EVAL': json_file,
                'ORACLE': [self.oracle] if self.oracle else None,
            })
            results = eval_tao(*eval_cfgs)
        return results
