# Data Preparation for TAO
There are four components to loading the TAO dataset into detectron2.

## Dataset 
`dataset.tao.load_tao_json` loads info into memory
- Input: annotation file path
- Output: `List[Dict]` - list of videos (`video_dict`)
    - `video_id`
    - `images`
        - `annotations`
            - `{bbox, bbox_mode, category_id, track_id, instance_id}`
        - `{file_name, height, width, not_exhaustive_category_ids, neg_category_ids,
            image_id, frame_id, video_id}`
    - `dataset_source`

`register_tao_instance` registers all the splits

## Loader
`video_dataset_loader.get_video_dataset_dicts` reads dataset name from 
registry
- Input: `dataset_names`
- Output:  `List[Dict]` same as above

## Mapper
`video_dataset_mapper.VideoDatasetMapper.__call__` samples frames from
`video_dict`, loads frames into memory, applies transforms on the images
and annotations, and wraps annotations into `Instances`. 
- Input: `video_dict` (single video info)
- Output: `List[Dict]` (list of images)
    - `images`
    - `instances`
    - `{file_name, height, width, not_exhaustive_category_ids, neg_category_ids,
            image_id, frame_id, video_id}`

## DataLoader
`video_dataset_dataloader.build_video_train_loader` creates dataloader
The dataloader uses the following collate_fn `lambda x: x[0]`.

A standard image batch: NCHW.

A standard video batch: NTCHW.

This one forces N=1: TCHW.
 
**TODO impl batch_size>1**  
- Input:
    - `cfg`
        - `DATASETS`
            - `TRAIN`
        - `DATALOADER`
            - `FILTER_EMPTY_ANNOTATIONS`
            - `SAMPLER_TRAIN`
            - `REPEAT_THRESHOLD`
            - `NUM_WORKERS`
        - `SOLVER`
            - `IMS_PER_BATCH`
    - `mapper`
- Output: `torch.nn.DataLoader`
 
