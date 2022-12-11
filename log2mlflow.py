import pandas as pd
import argparse
import json
import csv
import os
import mlflow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default=None, type=str)
    args = parser.parse_args()
    # path = "pus1.2_adaptive_seg_deit_small_patch16_224_mask_lr3e-4_WeakTrCOCOPseudoMask"
    path = args.path
    try:
        mlflow.get_run(path)
    except:
        mlflow.start_run(run_name=path)
    log_path = os.path.join(path, "log.txt")
    log_dict = []
    with open(log_path, "r") as f:
        lines = f.readlines()
        for r in lines:
            log_stats = json.loads(r)
            for key, value in log_stats.items():
                mlflow.log_metric(key, value, log_stats['epoch'])
            # log_dict.append(dict)
    # keys = log_dict[0].keys()
    # csv_file = path + "log_csv.csv"
    # with open(csv_file, 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=keys)
    #     writer.writeheader()
    #     for data in log_dict:
    #         writer.writerow(data)
