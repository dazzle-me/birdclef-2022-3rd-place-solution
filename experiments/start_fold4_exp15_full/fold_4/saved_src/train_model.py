import yaml
from trainer import Trainer
from os.path import join

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
                    max_snr_in_db=3,
                    p=0.5,
                ),
                AA.AddBackgroundNoise(
                    sounds_path=join(background_noise_root, "train_soundscapes/nocall"),
                    min_snr_in_db=0,
                    max_snr_in_db=3,
                    p=0.25,
                ),
                AA.AddBackgroundNoise(
                    sounds_path=join(background_noise_root, "aicrowd2020_noise_30sec/noise_30sec"),
                    min_snr_in_db=0,
                    max_snr_in_db=3,
                    p=0.25,
                ),
        ])
    cfg['training']['train_augs'] = AA.Compose(*augs) if augs != [] else AA.Compose([])
    # trainer = Trainer(cfg)
    # trainer.train()

    cfg['training']['train_folds'] = [0, 1, 2, 3, 4]
    cfg['general']['exp_name'] = f"{cfg['general']['exp_name']}_full"
    trainer = Trainer(cfg)
    trainer.train()
