""" Based off TrackEval/scripts/run_tao.py

Run example:
eval_tao.py --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL Tracktor++

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/tao/tao_training'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/tao/tao_training'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val'
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support
import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)-8s][%(process)d][%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
fp = Path(__file__).parent / 'TrackEval'
sys.path.insert(0, str(fp.absolute()))
import trackeval  # noqa: E402


def setup_argparse():
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.TAO.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    return eval_config, dataset_config, metrics_config


def setup_cfg(cfg):
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.TAO.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    config.update(cfg)

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    return eval_config, dataset_config, metrics_config


def postprocess_eval(ret, dataset, config):
    metrics = ret['TAO'][config['TRACKERS_TO_EVAL'][0]]

    to_take_mean = ['DetRe', 'DetPr', 'DetA', 'AssRe', 'AssPr', 'AssA', 'HOTA']
    rows = []
    for vid_id in metrics:
        for cls in metrics[vid_id]:
            hota_vid = metrics[vid_id][cls]['HOTA']
            if vid_id == 'COMBINED_SEQ' or 'GT-RHOTA_mean' not in hota_vid:
                continue
            for i in range(len(hota_vid['raw_gt_ids'])):
                rows.append({
                    'vid_name': vid_id,
                    'cls': cls,
                    'gt_track_id': hota_vid['raw_gt_ids'][i],
                    'detre': hota_vid['GT-DetRe_mean'][i],
                    'rhota': hota_vid['GT-RHOTA_mean'].tolist()[i],
                    'assa': hota_vid['GT-AssA_mean'].tolist()[i],
                    **{f'Vid_{m}_mean': hota_vid[m].mean() for m in to_take_mean}
                })
    fn = 'per_gt'
    if config['ORACLE']:
        oracle_type = config['ORACLE'][0]
        fn += f'_{oracle_type}_oracle'
    fp = dataset.get_output_fol(dataset.get_eval_info()[0][0]) + f'{fn}.csv'
    logging.info(f'Saving results to {fp}')
    pd.DataFrame(rows).to_csv(fp, index=False)


def eval_tao(eval_config, dataset_config, metrics_config):
    evaluator = trackeval.Evaluator(eval_config)
    logging.info('Start loading TAO dataset')
    dataset_list = [trackeval.datasets.TAO(dataset_config)]
    logging.info('Finish loading TAO dataset')
    metrics_list = []
    for metric in [trackeval.metrics.TrackMAP, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                   trackeval.metrics.HOTA]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    ret, msg = evaluator.evaluate(dataset_list, metrics_list)
    full_config = {**eval_config, **dataset_config, **metrics_config}
    postprocess_eval(ret, dataset_list[0], full_config)
    return ret


if __name__ == '__main__':
    eval_config, dataset_config, metrics_config = setup_argparse()
    eval_tao(eval_config, dataset_config, metrics_config)

