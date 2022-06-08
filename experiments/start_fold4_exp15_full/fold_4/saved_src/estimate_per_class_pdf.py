import os
from os.path import join
from argparse import ArgumentParser

import json
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

scored_birds = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", \
                "hawama", "hawcre", "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", \
                "omao", "puaioh", "skylar", "warwhe1", "yefcan"]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='./experiments')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--bins', type=int, default=10, help="Number of bins passed to plt.hist")
    args = parser.parse_args()

    root = args.root
    exp = args.exp
    fold = args.fold
    epoch = args.epoch
    bins = args.bins

    save_dir = join(root, exp, f"fold_{fold}/val_output/epoch_{epoch}_artifacts/")
    os.makedirs(save_dir, exist_ok=True)
    
    prediction_file = open(join(root, exp, f"fold_{fold}/val_output/epoch_{epoch}_validation.json"))
    predictions = json.load(prediction_file)

    birds_meta_file = open(join(root, exp, f"fold_{fold}/birds_meta.json"))
    birds_meta = json.load(birds_meta_file)

    probabilities = np.array(predictions['predictions'])
    labels = np.array(predictions['targets'])
        
    for bird in tqdm(scored_birds):
        bird_idx = birds_meta[bird]

        mask_true = labels[:, bird_idx] == 1.0
        mask_false = labels[:, bird_idx] != 1.0

        bird_positive_probs = probabilities[mask_true, bird_idx]
        bird_negative_probs = probabilities[mask_false, bird_idx]

        plt.figure(figsize=(12, 12))
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.title(f"Histogram of probabilities for {bird} on fold : {fold}")
        plt.hist(bird_positive_probs, bins=bins)
        plt.hist(bird_negative_probs, bins=bins, alpha=0.5)
        plt.xlim(0, 1.0)
        plt.yscale('log')
        plt.legend(["Positive probs", "Negative probs"])
        
        plt.savefig(fname=join(save_dir, f'{bird}_hist.png'))