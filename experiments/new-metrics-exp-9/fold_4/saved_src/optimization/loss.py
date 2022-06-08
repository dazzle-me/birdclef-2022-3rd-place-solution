from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import focal_loss
import torch
from torch.nn import BCEWithLogitsLoss
class LossFactory():
    def __init__(self):
        self.supported_losses = {
            'CrossEntropyLoss' : CrossEntropyLoss,
            'FocalLoss' : FocalLoss,
            'ArcFace_and_Linear' : custom_loss_1,
            'BCEWithLogitsLoss' : BCEWithLogitsLoss_class,
            'FocalBCEWithLogits' : FocalBCEWithLogits
            # 'MaskedBCEWithLogitsLoss' : MaskedBCEWithLogitsLoss
        }

    def get_loss(self, cfg, mode):
        loss_name = cfg['training']['train_loss'] if mode == 'train' else cfg['training']['val_loss']
        if loss_name in self.supported_losses.keys():
            return self.supported_losses[loss_name](cfg, mode)
        else:
            raise ValueError(f"Loss : {loss_name} is not supported")

# class MaskedBCEWithLogitsLoss():
#     def __init__(self, cfg, input=None, output=None):
#         self.name = 'BCEWithLogitsLoss'

#         params = cfg['loss']['BCEWithLogitsLoss']

#         self.input = input if input else params['input']
#         self.output = output if output else params['output']
#         self.reduction = params['reduction']
#         self.num_classes = cfg['model']['num_classes']

#     def __call__(self, input_dict, dict_data):
#         input = input_dict[self.input]
#         target = input_dict[self.output]
#         weight = input_dict['weight']

#         loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
#         loss = (loss.mean(dim=1) * weight) / weight.sum()
#         loss = loss.sum()
#         return loss

class FocalBCEWithLogits():
    def __init__(self, cfg, mode, *, input=None, output=None):
        self.name = 'FocalBCEWithLogits'

        params = cfg['loss']['FocalBCEWithLogits']

        self.input = input if input else params['input']
        self.output = output if output else params['output']
        
        self.reduction = params['reduction']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.num_classes = cfg['model']['num_classes']
        if mode == 'val':
            self.scored_birds_mask = torch.zeros(self.num_classes).to(cfg['general']['device'])
            scored_birds = cfg['general']['scored_birds']
            for index in scored_birds:
                self.scored_birds_mask[index] = 1
        self.mode = mode

    def __call__(self, input_dict, dict_data):
        input = input_dict[self.input]
        target = input_dict[self.output]
        weight = input_dict['weight'].squeeze()
        
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        probas = torch.sigmoid(input)
        loss = target * self.alpha * (1 - probas) ** self.gamma * bce_loss + \
            (1 - target) * probas ** self.gamma * bce_loss
        
        loss = (loss.mean(dim=1) * weight) / weight.sum()
        loss = loss.sum()
        return 10 * loss

class BCEWithLogitsLoss_class():
    def __init__(self, cfg, mode, *, input=None, output=None):
        self.name = 'BCEWithLogitsLoss'
        self.cfg = cfg
        params = cfg['loss']['BCEWithLogitsLoss']

        self.input = input if input else params['input']
        self.output = output if output else params['output']
        self.reduction = params['reduction']
        self.num_classes = cfg['model']['num_classes']

        ## scale scored birds to force model to learn them
        self.scored_scale = torch.ones(self.num_classes).to(cfg['general']['device'])
        scored_birds = cfg['general']['scored_birds']
        for index in scored_birds:
            self.scored_scale[index] = self.cfg['training']['scored_birds_scale']

        if mode == 'val':
            self.scored_birds_mask = torch.zeros(self.num_classes).to(cfg['general']['device'])
            scored_birds = cfg['general']['scored_birds']
            for index in scored_birds:
                self.scored_birds_mask[index] = 1
        self.mode = mode

    def __call__(self, input_dict, dict_data):
        input = input_dict[self.input]
        target = input_dict[self.output]
        weight = input_dict['weight'].squeeze()

        loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)

        ## mask loss for non-scored birds
        if self.mode == 'val':
            loss *= self.scored_birds_mask * (self.num_classes / 21)
        
        ## scale loss for scored birds
        loss *= self.scored_scale

        loss = (loss.mean(dim=1) * weight) / weight.sum()
        loss = loss.sum()
        return 10 * loss

class CrossEntropyLoss():
    def __init__(self, cfg, input=None, output=None):
        self.name = 'CrossEntropyLoss'

        params = cfg['loss']['crossentropy']

        self.input = input if input else params['input']
        self.output = output if output else params['output']

        self.num_classes = cfg['model']['num_classes']

    def __call__(self, input_dict, dict_data):
        input = input_dict[self.input]
        target = dict_data[self.output].squeeze()
        return F.cross_entropy(input, target)

class FocalLoss():
    def __init__(self, cfg, input=None, output=None):
        self.name = 'FocalLoss'

        params = cfg['loss']['focal']

        self.input = input if input else params['input']
        self.output = output if output else params['output']
        self.gamma = params['gamma']

        self.num_classes = cfg['model']['num_classes']

    def __call__(self, input_dict, dict_data):
        input = input_dict[self.input]
        target = dict_data[self.output].squeeze()

        return focal_loss(input, target, gamma=self.gamma, alpha=0.5, reduction='mean')

class custom_loss_1():
    def __init__(self, cfg):
        self.name = 'Focal_binary_head_arcface_head'
        self.margin_loss = FocalLoss(cfg, input='margin_head', output='id')
        self.binary_loss = FocalLoss(cfg, input='binary_head', output='id')

    def __call__(self, preds, data):
        return self.margin_loss(preds, data) + self.binary_loss(preds, data)
