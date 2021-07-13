
""" run_kitti.py

Run example:
run_kitti.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL CIWT

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
        'GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box_train'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/kitti/kitti_2d_box_train/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['car', 'pedestrian'],  # Valid: ['car', 'pedestrian']
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val', 'training_minus_val', 'test'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': ''  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['Hota','Clear', 'ID', 'Count']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)-8s][%(process)d][%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == '__main__':
    cmd = 'python ' + ' '.join(sys.argv)
    logging.info(cmd)
    freeze_support()

    # Command line interface:
    my_config = {'UUID': 'default_uuid', 'SUFFIX': ''}
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config, **my_config}  # Merge default configs
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

    import wandb
    for i in range(10):
        try:
            wandb_run = wandb.init(project='object-motion', resume='allow', id=config['UUID'], dir='/scratch/cluster/jozhang/logs', save_code=True)
        except wandb.errors.error.UsageError as e:
            # see https://github.com/wandb/client/issues/1409#issuecomment-723371808
            if i == 9:
                logging.error(f'Could not init wandb in 10 attempts, exiting')
                raise e
            logging.warning(f'wandb.init failed {i}th attempt, retrying')
            import time
            time.sleep(10)
    wandb.config.update({f'new-eval-{k}': v for k, v in config.items()}, allow_val_change=True)

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    ret, msg = evaluator.evaluate(dataset_list, metrics_list)
    '''
    Dataset >> Trackers >> Episode >> Class >> Metrics
    '''
    metrics = ret['Kitti2DBox'][dataset_config['TRACKERS_TO_EVAL'][0]]

    import numpy as np
    def get_worst_best(met, k):
        met_ids = np.argsort(met)
        return met_ids[:k], met_ids[-k:]
    rows = []
    for vid_id in metrics:
        hota_vid = metrics[vid_id]['car']['HOTA']
        if vid_id == 'COMBINED_SEQ' or 'GT-RHOTA_mean' not in hota_vid:
            continue
        rhota_easy_ids, rhota_hard_ids = get_worst_best(hota_vid['GT-RHOTA_mean'], 5)
        gt_assa_easy_ids, gt_assa_hard_ids = get_worst_best(hota_vid['GT-AssA_mean'], 5)
        pr_assa_easy_ids, pr_assa_hard_ids = get_worst_best(hota_vid['PR-AssA_mean'], 10)
        rows.append([vid_id, 'easy', 'gt', *rhota_easy_ids, *gt_assa_easy_ids])
        rows.append([vid_id, 'hard', 'gt', *rhota_hard_ids, *gt_assa_hard_ids])
        rows.append([vid_id, 'easy', 'pr', *pr_assa_easy_ids])
        rows.append([vid_id, 'hard', 'pr', *pr_assa_hard_ids])

    import pandas as pd
    header = ['vid_id', 'difficulty', 'type', *[str(i) for i in range(10)]]
    fp = dataset_list[0].get_output_fol(dataset_list[0].get_eval_info()[0][0]) + 'tids.csv'
    pd.DataFrame(rows).to_csv(fp, header=header, index=False)

    per_vid_hota = ['AssA']
    per_vid_mota = ['IDSW']
    per_vid_log = {}
    for vid_id in metrics:
        metrics_vid = metrics[vid_id]['car']
        for m in per_vid_hota:
            per_vid_log[f'Vid{vid_id}-{m}_mean'] = metrics_vid['HOTA'][m].mean()
        for m in per_vid_mota:
            per_vid_log[f'Vid{vid_id}-{m}'] = metrics_vid['CLEAR'][m]

    car_metrics = metrics['COMBINED_SEQ']['car']
    to_take_mean = ['DetRe', 'DetPr', 'DetA', 'AssRe', 'AssPr', 'AssA', 'HOTA']
    for m in to_take_mean:
        car_metrics['HOTA'][f'{m}_mean'] = car_metrics['HOTA'][m].mean()

    hota_keep = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)'] + [f'{m}_mean' for m in to_take_mean]
    car_hota = {f'HOTA-{k}': v for k, v in car_metrics['HOTA'].items() if k in hota_keep}
    car_clear = {f'CLEAR-{k}': v for k, v in car_metrics['CLEAR'].items()}
    car_iden = {f'IDENTITY-{k}': v for k, v in car_metrics['Identity'].items()}
    to_log = {**car_clear, **car_hota, **car_iden, **per_vid_log}
    to_log = {f'{k}{config["SUFFIX"]}': v for k, v in to_log.items()}
    for k, v in to_log.items():
        wandb.run.summary[k] = v
    wandb.log(to_log)
    wandb_run.finish()
    for k, v in per_vid_log.items():
        logging.info(f' {k} - {v}')
    for k, v in car_hota.items():
        logging.info(f' {k} - {v}')
    logging.info(f'MOTA - {car_clear["CLEAR-MOTA"]}, IDSW {car_clear["CLEAR-IDSW"]}, Frag {car_clear["CLEAR-Frag"]}')
    logging.info(f'we are done')
