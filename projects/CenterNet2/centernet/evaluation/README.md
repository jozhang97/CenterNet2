# Evaluating Multi-Object Tracking (MOT)

There are 4 components to evaluation.
The pipeline is implemented in `train_net.py`

`def do_test()`
- `video_file_name` (start)
- `video_tensor` (dataloader)
- `instances`    (model)
- `results.json` (evaluator)
- `summary.txt`  (TrackEval)

## Dataloader
`data/video_dataset_dataloader.py` 
- `build_video_test_loader` produces a single GPU dataloader for evaluation.

## Model
`modeling/meta_arch/base_video_rcnn.py`
- `BaseVideoRCNN.inference` produce tracks from videos
by mapping each detection to its own track.


## Evaluator
`evaluation/mot_evaluation.py`
- `MOTEvaluator` As we iterate over test set, evaluator will `process` the 
inputs and outputs to save the predictions into 
test format.
`_eval_predictions` saves the predictions to disk
and calls TrackEval methods to perform evaluation.

## TrackEval
`eval_tao.py`
- `setup_cfg` prepares configs
- `eval_tao` reads tracker predictions and GT annotations from disk
to perform evaluation. 

# Notes
- Performance is 1-2pts lower than earlier setup
- `INPUT.TEST_SIZE` is important. Setting this to 640
    reduces performance significantly (bug?), but hopefully
    ordering is the same. The TITAN machines do not
    have enough GPU memory for full image sizes.