from multiprocessing import dummy
import torch
import torch.nn as nn

import torchaudio as ta
from torchaudio.functional import spectrogram

class MelSpecRandomPower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        augs = cfg['training']['augs']['random_power']
        self.cfg = cfg

        self.left = augs['left']
        self.right = augs['right']
        self.scale = self.right - self.left
        self.mel_scale = ta.transforms.MelScale(
            self.cfg['data']['mel_bins'],
            self.cfg['data']['sample_rate'],
            self.cfg['data']['fmin'],
            self.cfg['data']['fmax'],
            cfg['data']['window_size'] // 2 + 1,
            norm=None
        )
        window = torch.hann_window(self.cfg['data']['window_size'])
        self.register_buffer("window", window)
    @torch.no_grad()
    def __call__(self, audio):
        power = torch.rand(1).item()
        power = self.left + (self.scale * power)
        # print(power)
        spec = spectrogram(
            audio, 
            n_fft=self.cfg['data']['window_size'],
            win_length=self.cfg['data']['window_size'],
            hop_length=self.cfg['data']['hop_size'],
            window=self.window,
            pad=0,
            power=power,
            normalized=False,
        )
        mel_scale = self.mel_scale(spec)
        return mel_scale

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('/home/eduard/kaggle/birdclef-2022/server_src/config/base_config.yaml'))
    cfg['training']['augs']['random_power']['left'] = 2.0
    cfg['training']['augs']['random_power']['right'] = 2.0
    transformer = MelSpecRandomPower(cfg)
    transformer_orig = ta.transforms.MelSpectrogram(
            sample_rate=cfg['data']['sample_rate'],
            n_fft=cfg['data']['window_size'],
            win_length=cfg['data']['window_size'],
            hop_length=cfg['data']['hop_size'],
            f_min=cfg['data']['fmin'],
            f_max=cfg['data']['fmax'],
            pad=0,
            n_mels=cfg['data']['mel_bins'],
            power=cfg['data']['power'],
            normalized=False,
        )
    dummy_audio = torch.randn((2, 32000))
    
    for i in range(10):
        random_power_spec = transformer(dummy_audio)
        orig_spec = transformer_orig(dummy_audio)
        print(torch.allclose(random_power_spec, orig_spec))

    cfg['training']['augs']['random_power']['left'] = 0.5
    cfg['training']['augs']['random_power']['right'] = 3.5
    transformer = MelSpecRandomPower(cfg)
    
    for i in range(10):
        random_power_spec = transformer(dummy_audio)
        orig_spec = transformer_orig(dummy_audio)
        print(torch.allclose(random_power_spec, orig_spec))

    