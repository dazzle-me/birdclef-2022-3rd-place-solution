import torch
import torch.nn as nn

from models.layers.arcface import MagrginLinear, ArcMarginProduct

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

class MarginHead(nn.Module):
    def __init__(self, num_class=15578, emb_size = 1280, s=64., m=0.5):
        super(MarginHead,self).__init__()
        self.fc = MagrginLinear(embedding_size=emb_size, classnum=num_class , s=s, m=m)

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea)
        return logit

class ArcmarginHead(nn.Module):
    def __init__(self, num_class=15578, emb_size = 1280, s=64., m=0.5, ls_eps=0.0, easy_margin=False):
        super(ArcmarginHead,self).__init__()
        self.fc = ArcMarginProduct(s=s, m=m, in_features=emb_size, out_features=num_class, easy_margin=easy_margin, ls_eps=ls_eps)

    def forward(self, data):
        image, label = data
        logit = self.fc(image, label)
        return logit

