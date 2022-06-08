import yaml
from trainer import Trainer
from os.path import join
from pprint import pprint

import audiomentations as AA
if __name__ == '__main__':
    cfg = yaml.safe_load(open('./config/base_config.yaml', 'r'))
    augs = []
    if cfg['training']['augs']['add_background_noise']:
        background_noise_root = '/workspace/datasets/bird-clef/no_call_2021'
        augs.append([
                AA.AddBackgroundNoise(
                    sounds_path=join(background_noise_root, "ff1010bird_nocall/nocall"),
                    min_snr_in_db=0,
                    max_snr_in_db=40,
                    p=0.5,
                ),
                AA.AddBackgroundNoise(
                    sounds_path=join(background_noise_root, "train_soundscapes/nocall"),
                    min_snr_in_db=0,
                    max_snr_in_db=40,
                    p=0.5,
                ),
                AA.AddBackgroundNoise(
                    sounds_path=join(background_noise_root, "aicrowd2020_noise_30sec/noise_30sec"),
                    min_snr_in_db=0,
                    max_snr_in_db=40,
                    p=0.5,
                ),
        ])
    params = cfg['training']['augs']['gaussian_noise']
    augs.append([AA.AddGaussianSNR(
        min_snr_in_db=params['min_snr'],
        max_snr_in_db=params['max_snr'],
        p=params['proba']
    )])
    augs = [item for sublist in augs for item in sublist]
    cfg['training']['train_augs'] = AA.Compose(augs) if augs != [] else AA.Compose([])
    pprint(cfg)
    trainer = Trainer(cfg)
    trainer.train()

    cfg['training']['train_folds'] = [0, 1, 2, 3, 4]
    cfg['general']['exp_name'] = f"{cfg['general']['exp_name']}_full"
    trainer = Trainer(cfg)
    trainer.train()
