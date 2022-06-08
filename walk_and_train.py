import os
from os.path import join, abspath
from glob import glob
import subprocess
import argparse

config_mapping = {
    "new-metrics-exp-23" : "./config/tf_effnet_v2_s_in21k.yaml",
}

import shutil
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='/workspace/bclf/reproduced_experiments')
    parser.add_argument('--exp-dir', type=str, default='experiments')

    args = parser.parse_args()

    base_path = os.getcwd()
    dump_dir = args.exp_dir
    save_dir = args.save_dir

    train_fold = 4
    # experiments_dirs = sorted(os.listdir(dump_dir))
    experiments_dirs = [
        # 'new-metrics-exp-9',
        # 'new-metrics-exp-17',
        # 'new-metrics-exp-18',
        'new-metrics-exp-19',
        # 'new-metrics-exp-23',
        # 'start_fold4_exp13_full_r',
        # 'start_fold4_exp15_full',
        # 'start_fold4_exp17_full'
    ]
    for experiment in experiments_dirs:
        experiment_path = join(base_path, dump_dir, experiment, f"fold_{train_fold}", 'saved_src')
        os.chdir(experiment_path)
        cwd = os.getcwd()
        print(experiment_path)
        if experiment == "new-metrics-exp-23": ## specific config flag is required for that one experiment
            subprocess.run(f'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train_model.py --cfg {config_mapping[experiment]}', shell=True)
        else:
            subprocess.run(f'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train_model.py', shell=True)
        inner_exp_dir = f'{experiment_path}/experiments'
        if os.path.isdir(inner_exp_dir):
            copytree(inner_exp_dir, save_dir)
