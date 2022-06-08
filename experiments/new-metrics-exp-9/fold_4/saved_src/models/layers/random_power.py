import torch
import torch.nn as nn

class RandomPower(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.left_border = cfg['training']['augs']['random_power']['left']
        self.right_border = cfg['training']['augs']['random_power']['right']

    @torch.no_grad()
    def __call__(self, spectrograms):
        '''
            input: 
                :param: spectorgrams - torch.Tensor of shape (batch_size, num_channels, n_mels, time)
            retrun:
                spectrograms - same Tensor, but raised to random power defined by 
                "left_border" and "right_border"
        '''
        bs = spectrograms.shape[0]
        power = torch.randn(bs).to(spectrograms.device)
        power = self.left_border + power * (self.right_border - self.left_border)
        if len(spectrograms.shape) == 3:
            power = power.view(-1, 1, 1)
        elif len(spectrograms.shape) == 4:
            power = power.view(-1, 1, 1, 1)
        # print(spectrograms.shape, power.shape, power)
        spectrograms = torch.pow(spectrograms, power)
        return spectrograms