import torch
from torch import dtype
import torch.nn as nn 

class ConcatMix(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg['general']['device']
    
    @torch.no_grad()
    def forward(self, audios, labels, weights=None):
        '''
            input:
                :param: audios, - torch.Tensor of shape (batch_size, length)
                :param: labels - torch.Tensor of shape (batch_size, num_classes)
                :param: weights - torch.Tensor of shape (batch_size, )
            output:
                :param: cat_audios - torch.Tensor of shape (batch_size, 2 * length)
                :param: new_labels - torch.Tensor of shape (batch_size, num_classes)
                :param: new_weights - torch.Tensor of shape (batch_size, )

            This method performs concatenation of random pairs of audios from batch
            and also changes labels of given audio to the union of labels of pair.

            NB: works for n >= 2
        '''
        bs = audios.shape[0]
        
        if bs >= 3:
            shifts = torch.randint(1, bs - 1, size=(1, )).item()
        else:
            shifts = 1
        perm = torch.roll(torch.arange(0, bs), shifts=shifts).long()
        # print(f"permutation : {perm}")
        shuffled_audios = audios[perm].to(self.device)
        shuffled_labels = labels[perm].to(self.device)
        
        if weights is not None:
            shuffled_weights = weights[perm].to(self.device)

        cat_audios = torch.cat([audios, shuffled_audios], dim=1)
        new_labels = torch.clip(shuffled_labels + labels, min=0, max=1)
        
        if weights is not None:
            new_weights = (weights + shuffled_weights) / 2
            return cat_audios, new_labels, new_weights
        else:
            return cat_audios, new_labels


if __name__ == '__main__':
    ## 1 second audio given sr = 32'000
    batch_size = 3
    audio_length = 32000
    cfg = {
        'general' : {
            'device' : 'cuda:0'
        }
    }

    audio = torch.randn((batch_size, audio_length)).cuda()
    labels = torch.Tensor([
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0]
    ]).cuda()
    weights = torch.Tensor([
        [0.5],
        [1.0],
        [0.1]
    ]).cuda()

    mixer = ConcatMix(cfg)
    print(f"Before mixer : {audio.shape}, {labels.shape}, {weights.shape}")
    print(f"Labels : \n{labels}")
    print(f"Weights\n:{weights}")
    audio, labels, weights = mixer(audio, labels, weights)
    print(f"After mixer : {audio.shape}, {labels.shape}, {weights.shape}")
    print(f"Labels :\n{labels}")
    print(f"Weights\n:{weights}")