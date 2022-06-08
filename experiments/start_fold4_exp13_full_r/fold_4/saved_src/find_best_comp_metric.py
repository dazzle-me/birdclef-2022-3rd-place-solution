from argparse import ArgumentParser

import os
from os.path import join

import json
import numpy as np

from metrics.bird_metric import CompMetric
import matplotlib.pyplot as plt
from tqdm import tqdm

scored_birds = [153, 156, 157, 9, 44, 185, 47, 194, 196, 197, 64, 65, 199, 201, 202, 90, 219, 111, 239, 246, 253]

if __name__ == '__main__':
    parser = ArgumentParser()

    ## by default, the best model will be selected
    parser.add_argument('--exp-root', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--fold', type=int)

    args = parser.parse_args()

    ## delete this folder in the future
    exp_path = join(args.exp_root, args.experiment, f'fold_{args.fold}')

    best_weights = join(exp_path, 'weights/best_model.txt')
    with open(best_weights, 'r') as file:
        best_weights_path = file.readline()
    ## best_weights_path format : epoch_{epoch}_metric_{metric}.pth
    epoch = best_weights_path.split('_')[1]

    with open(join(exp_path, f'val_output/epoch_{epoch}_validation.json'), 'r') as file:
        data = json.load(file)

    predictions = np.array(data['predictions'])
    targets = np.array(data['targets'])

    print(predictions.shape, targets.shape)
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    import yaml
    cfg = yaml.safe_load(open(join(exp_path, 'saved_src', 'config/base_config.yaml'), 'r'))

    f_scores = []
    for threshold in tqdm(thresholds):
        cfg['metrics']['bird_metric']['threshold'] = threshold
        metric = CompMetric(cfg)
        metric.reset()
        metric.update(
            {
                'logits' : predictions,
            },
            {
                'target' : targets
            }
        )
        f_scores.append(metric.compute())

    plt.figure(figsize=(12, 12))
    plt.title("competition metric dependence on threshold")
    plt.ylabel("bird metric")
    plt.xlabel("threshold")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.plot(thresholds, f_scores)
    plt.grid(True)

    best_threshold_index = np.argmax(f_scores)
    best_threshold = list(thresholds)[best_threshold_index]

    plt.axvline(x=best_threshold, ymax=f_scores[best_threshold_index], color='red')
    plt.legend([f'best comp metric : {f_scores[best_threshold_index]:.3f}, best threshold : {best_threshold:.2f}'])
    plt.savefig(join(exp_path, 'val_output/comp_metric.png'))
    np.save(join(exp_path, 'val_output/comp_metrics.npy'), f_scores)
