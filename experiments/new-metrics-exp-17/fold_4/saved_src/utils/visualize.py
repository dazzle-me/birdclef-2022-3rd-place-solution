import matplotlib.pyplot as plt
import pandas as pd

import os
from os.path import join
import json

from PIL import Image

from argparse import ArgumentParser

if __name__ == '__main__':
    train_csv = '/home/eduard/kaggle/happy-whale-and-dolphin/data/detic-crop/train_indexed.csv'
    
    data_dir = '/home/eduard/kaggle/happy-whale-and-dolphin/data/detic-crop'
    
    train_image_dir = f'{data_dir}/cropped_train_images/cropped_train_images'
    test_image_dir = f'{data_dir}/cropped_test_images/cropped_test_images'

    parser = ArgumentParser()

    parser.add_argument('--exp-dir', type=str, default='./experiments/')
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, f'{args.exp_name}_stage_2', f'fold_{args.fold}')

    csv = pd.read_csv(train_csv)
    prediction_dict = json.load(open(join(exp_dir, 'prediction_dict.json'), 'r'))
    for test_image_name in prediction_dict.keys():
        for identity, probability in zip(prediction_dict[test_image_name]['identities'], prediction_dict[test_image_name]['probabilities']):
            csv_sample = csv[csv.individual_id == identity].reset_index(drop=True).iloc[:5]
            fig, ax = plt.subplots(ncols=min(5, len(csv_sample)) + 1, nrows=1, figsize=(32, 16))
            
            test_image = Image.open(join(test_image_dir, test_image_name))
            ax[0].imshow(test_image)
            ax[0].set_title("Test image")
            fig.suptitle(f'From 1:, - images of {identity}, probability : {probability:.4f}')
            for enum, image_name in enumerate(csv_sample['image']):
                image = Image.open(join(train_image_dir, image_name))
                ax[enum + 1].imshow(image)
            plt.show()
            


