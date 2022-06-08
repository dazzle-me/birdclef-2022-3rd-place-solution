import torch
import torch.nn as nn
import torchaudio as ta
import timm
from models.layers.gem import GeM
from models.layers.mixup import Mixup
from models.layers.concat_mix import ConcatMix
from models.layers.random_power_melspec import MelSpecRandomPower

from torch.cuda.amp import autocast
import numpy as np

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        self.noise_level = 0.05
        self.n_classes = cfg['model']['num_classes']
        self.mel_spec = MelSpecRandomPower(cfg)

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=None)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.backbone = timm.create_model(
            cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
            num_classes=0,
            global_pool="",
            in_chans=cfg['model']['in_chans'],
        )

        if "efficientnet" in cfg['model']['backbone']:
            backbone_out = self.backbone.num_features
        elif "rexnet" in cfg['model']['backbone']:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()

        self.head = nn.Linear(backbone_out, self.n_classes)

        if cfg['general']['weights_path'] is not None:
            sd = torch.load(cfg['general']['weights_path'], map_location="cpu")
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", cfg['general']['weights_path'])
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.factor = int(cfg['data']['crop_length'] / 5.0)
        self.mixup = Mixup(mix_beta=cfg['training']['augs']['mix_beta'])
        self.concat_mix = ConcatMix(cfg)
        self.attention = nn.Sequential(nn.Linear(backbone_out, 512), nn.ReLU(), nn.Linear(512, 1))

    def train_step(self, data):
        return self.forward(data)

    def forward(self, data):
        if self.is_train:
            x = data['audio']
            y = data['target']
            weight = data['weight']
            if np.random.random() <= self.cfg['training']['augs']['concat_mix']:
                x, y, weight = self.concat_mix(x, y, weight)
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            x = data['audio']
            y = data['target']
            weight = data['weight']
        with autocast(enabled=False):
            x = self.wav2img(x) ## (bs * parts, mel, time // parts)
            if self.cfg['data']['mel_norm']:
                # x = x/(x.max()+0.0000001)*(torch.rand(1).item()*1+0.5)*x.max()
                # x = self.amplitude_to_db(x)
                x = (x + 80) / 80
        if self.is_train:
            if np.random.random() <= self.cfg['training']['augs']['white_noise']:
                x = x + torch.randn(*x.shape).to(self.cfg['general']['device']) * x.mean() * self.noise_level * (torch.randn(1).item() + 0.3)
        # print(x.mean())
        x = x.permute(0, 2, 1) ## (bs * parts, time // parts, mel)
        x = x[:, None, :, :] ## (bs * parts, 1, time // parts, mel)
        if self.is_train:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3) ## (bs * parts, time // parts, 1, mel)
            x = x.reshape(b // self.factor, self.factor * t, c, f) ## (bs, time, 1, mel)
            if np.random.random() <= self.cfg['training']['augs']['mixup']:
                x, y, weight = self.mixup(x, y, weight)
            # if self.cfg.mixup2:
            #     x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)
        ## do linear gradient right before feeding into backbone
        ## x.shape == (bs * parts, 1, time // parts, mel)
        bs, channels, time, mels = x.shape
        if self.cfg['model']['use_coord_conv']:
            lin_grad = torch.ones((mels, time)) / torch.arange(1, time + 1)
            lin_grad = lin_grad.T.reshape(1, 1, time, mels).repeat_interleave(bs, dim=0).cuda()
            x = torch.cat([x, lin_grad], dim=1)
        x = self.backbone(x)
        if self.is_train:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)
        ## (batch, channles, time, frequency)
        x = x.mean(dim=3) ## (batch, channels, time)
        x = x.permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1) ## (batch, time)
        x2 = (x * attn_weights).sum(dim=1) ## (batch, time, features)
        logits = self.head(x2)        
        return {"logits": logits.sigmoid(), "logits_raw": logits, 'weight' : weight, 'target' : y}

class RCNN(nn.Module):
    def __init__(self, cfg):
        super(RCNN, self).__init__()
        self.cfg = cfg
        self.n_classes = cfg['model']['num_classes']
        self.mel_spec = ta.transforms.MelSpectrogram(
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

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=None)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.backbone = timm.create_model(
            cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
            num_classes=0,
            global_pool="",
            in_chans=cfg['model']['in_chans'],
        )

        if "efficientnet" in cfg['model']['backbone']:
            backbone_out = self.backbone.num_features
        elif "rexnet" in cfg['model']['backbone']:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()

        self.gru = nn.GRU(backbone_out, backbone_out, batch_first=True, bidirectional=True, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(2 * backbone_out, self.n_classes),
        )

        if cfg['general']['weights_path'] is not None:
            sd = torch.load(cfg['general']['weights_path'], map_location="cpu")
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", cfg['general']['weights_path'])
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.factor = int(cfg['data']['crop_length'] / 5.0)
        self.mixup = Mixup(mix_beta=cfg['training']['augs']['mix_beta'])

    def train_step(self, data):
        return self.forward(data)

    def forward(self, data):
        if self.is_train:
            x = data['audio']
            y = data['target']
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
            weight = data['weight']
        else:
            x = data['audio']
            y = data['target']
            weight = data['weight']
        with autocast(enabled=False):
            x = self.wav2img(x) ## (bs * parts, mel, time // parts)
            if self.cfg['data']['mel_norm']:
                x = (x + 80) / 80
        x = x.permute(0, 2, 1) ## (bs * parts, time // parts, mel)
        x = x[:, None, :, :] ## (bs * parts, 1, time // parts, mel)
        if self.is_train:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3) ## (bs * parts, time // parts, 1, mel)
            x = x.reshape(b // self.factor, self.factor * t, c, f) ## (bs, time, 1, mel)
            if np.random.random() <= self.cfg['training']['augs']['mixup']:
                x, y, weight = self.mixup(x, y, weight)
            # if self.cfg.mixup2:
            #     x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)
        ## do linear gradient right before feeding into backbone
        ## x.shape == (bs * parts, 1, time // parts, mel)
        bs, channels, time, mels = x.shape
        if self.cfg['model']['use_coord_conv']:
            lin_grad = torch.ones((mels, time)) / torch.arange(1, time + 1)
            lin_grad = lin_grad.T.reshape(1, 1, time, mels).repeat_interleave(bs, dim=0).cuda()
            x = torch.cat([x, lin_grad], dim=1)
        x = self.backbone(x)
        if self.is_train:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)
        ## (bs, channels, time, freq)
        x = x.mean(dim=3)
        x = x.permute(0, 2, 1)
        (x, _) = self.gru(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return {"logits": logits.sigmoid(), "logits_raw": logits, 'weight' : weight, 'target' : y}

