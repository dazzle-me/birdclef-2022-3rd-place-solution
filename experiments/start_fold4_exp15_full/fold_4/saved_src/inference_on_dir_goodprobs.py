import importlib
import os
from os.path import join, abspath
from cv2 import exp
import yaml
from argparse import ArgumentParser
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import product
import numpy as np

## new : [400, 402, 403, 405, 429, 431, 432, 440, 442, 443, 444, 445, 178, 448, 450, 464, 471, 479, 493, 500, 507]
bird_names = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", "hawama", "hawcre", \
    "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", "omao", "puaioh", "skylar", "warwhe1", "yefcan"]

def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.audio_dir = test_dir
        self.audio_list = os.listdir(test_dir)
        
        self.num_parts = 12
        self.duration = 5
        
    def __getitem__(self, index):
        audio_name = self.audio_list[index]
        ## load whole audio
        audio, sr = librosa.load(join(self.audio_dir, audio_name), sr=None, offset=0)
        
        sr = 32000
        ## not a whole n-second audio
        if audio.shape[0] % sr != 0:
            audio = np.pad(audio, (0, sr - audio.shape[0] % sr))
        
        expected_length = 1 ## in seconds
        assert len(audio == sr * expected_length)
        audio = torch.Tensor(audio)
        audio = audio.reshape(self.num_parts, len(audio) // self.num_parts) 
        return {
            'audio' : torch.Tensor(audio),
            'filename' : audio_name
        }
    def __len__(self) -> int:
        return len(self.audio_list)

if __name__ == '__main__':
    device = 'cuda:0'

    parser = ArgumentParser()

    parser.add_argument('--experiment-dir', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--inference-dir')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    print(args)
    import sys
    src_path = abspath(join(args.experiment_dir, args.exp, 'fold_0/saved_src'))
    
    cfg = yaml.safe_load(open(join(src_path, 'config/base_config.yaml')))
    scored_birds = cfg['general']['scored_birds']
    
    cfg['model']['pretrained'] = False
    sys.path.insert(0, join(src_path, 'models'))
    sys.path.insert(0, src_path)
    
    model = importlib.import_module('model').Net(cfg).eval().to(device)
    with open(join(args.experiment_dir, args.exp, 'fold_0/weights/best_model.txt')) as file:
        best_model_name = file.readline()
    weights_path = torch.load(join(args.experiment_dir, args.exp, 'fold_0/weights', best_model_name))['model']
    status = model.load_state_dict(weights_path)
    if status:
        print("Weights loaded successfully")
    else:
        print(f"Wrong weights, path : {weights_path}")
    model.is_train = False

    dataset = TestDataset(args.inference_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    pred = {'row_id' : [], 'target' : [], 'score' : []}
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = batch_to_device(data, device)
            ## data['audio'].shape == (bs, num_parts, sr * 5) -> 
            ## data['audio'].shape == (bs * num_parts, sr * 5) ## inference on 5 sec crops
            bs, num_parts, time = data['audio'].shape
            data['audio'] = data['audio'].reshape(bs * num_parts, time)
            prediction = model({
                'audio' : data['audio'],
                'target' : -1,
                'weight' : -1
            })['logits'].detach().cpu().numpy()
            
            ## prediction for every offset, for every bird
            prediction = prediction.reshape(bs, num_parts, -1)
            for batch_idx, (filename, score) in enumerate(zip(data['filename'],  prediction)):
                soundscape_name = filename.split('.')[0]
                for offset_index, offset in enumerate(range(0, 60, 5)):
                    for bird, index in zip(bird_names, scored_birds):
                        pred_name = f'{soundscape_name}_{bird}_{offset + 5}'
                        pred['row_id'].append(pred_name)
                        pred['target'].append(True if score[offset_index][index] > args.threshold else False)
                        pred['score'].append(score[offset_index][index])
    results = pd.DataFrame(pred, columns=['row_id', 'target', 'score'])
    results.drop(columns=['score']).to_csv(join(args.experiment_dir, args.exp, 'submission.csv'), index=False)
    results.to_csv(join(args.experiment_dir, args.exp, 'submission_w_probs.csv'), index=False)