import torch
import torch.nn as nn

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

class BinaryHead(nn.Module):
    def __init__(self, num_class=15578, emb_size = 1280, s = 16.0):
        super(BinaryHead,self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea)*self.s
        return logit