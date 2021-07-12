from detectron2.data.catalog import DatasetCatalog
import itertools

def get_video_dataset_dicts(
    dataset_names, filter_empty=True, gen_inst_id=False,
):
    '''
    :return dataset_dicts: {video_id, images: List[Dict], dataset_source}
    images: {file_name, height, width, not_exhaustive_category_ids, neg_category_ids,
            image_id, frame_id, video_id, annotations: List[Dict]}
    annotations: {bbox, bbox_mode, category_id, track_id, instance_id}
    '''
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    inst_count = 1000000
    video_datasets = []
    for source_id, (dataset_name, dicts) in enumerate(
        zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        videos = {}
        single_video_id = 1000000
        for image in dicts:
            video_id = image.get('video_id', -1)
            if video_id == -1:
                single_video_id = single_video_id + 1
                video_id = single_video_id
            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id, 'images': [],
                    'dataset_source': source_id}
            if gen_inst_id:
                for x in image['annotations']:
                    if 'instance_id' not in x:
                        inst_count += 1
                        x['instance_id'] = inst_count
            videos[video_id]['images'].append(image)
        video_datasets.append([v for v in videos.values()])
    video_datasets = list(itertools.chain.from_iterable(video_datasets))
    return video_datasets
