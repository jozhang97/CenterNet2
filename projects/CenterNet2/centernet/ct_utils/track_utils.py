from collections import defaultdict
import itertools

def _make_track_ids_unique(annotations):
    """
    Makes the track IDs unqiue over the whole annotation set. Adapted from https://github.com/TAO-Dataset/
    :param annotations: the annotation set
    :return: the number of updated IDs
    """
    track_id_videos = {}
    track_ids_to_update = set()
    max_track_id = 0
    for ann in annotations:
        t = ann['track_id']
        if t not in track_id_videos:
            track_id_videos[t] = ann['video_id']

        if ann['video_id'] != track_id_videos[t]:
            # Track id is assigned to multiple videos
            track_ids_to_update.add(t)
        max_track_id = max(max_track_id, t)

    if track_ids_to_update:
        next_id = itertools.count(max_track_id + 1)
        new_track_ids = defaultdict(lambda: next(next_id))
        for ann in annotations:
            t = ann['track_id']
            v = ann['video_id']
            if t in track_ids_to_update:
                ann['track_id'] = new_track_ids[t, v]
    return len(track_ids_to_update)

def _make_track_ids_smallest(annotations):
    """
    Adapted from motion3d:scripts/gen_pseudo_gt_tracks.py
    Maps track_ids to smallest set of natural numbers
    """
    tids = set(a['track_id'] for a in annotations)
    tid2nat = {tid: i for i, tid in enumerate(sorted(tids))}

    for a in annotations:
        a['track_id'] = tid2nat[a['track_id']]
    return max(tid2nat.values())

