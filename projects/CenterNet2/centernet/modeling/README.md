# Model
## ReID

Data
- video_dataloader provides `gt_instances.gt_track_ids` 
- track ids are unique across videos and smallest

`centernet_head.py`
- produces `reid_hm`

`centernet.py`
- `_get_ground_truth` returns tids w/corresponding levels
- `loss` consumes `tids`, `reid_hm` and penalizes embedding distances
- `predict_single_level` produces instances with reid

# Association method
Things to try (TODO)
- ReID distance
- IoU distance
- Kalman filter
- Center distance
