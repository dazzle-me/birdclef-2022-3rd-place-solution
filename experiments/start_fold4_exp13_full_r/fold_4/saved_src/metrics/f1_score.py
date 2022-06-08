import sklearn
import torch
from sklearn.metrics import f1_score
import numpy as np

class F1():
    def __init__(self, cfg, input=None, output=None):
        params = cfg['metrics']['f1']

        self.input = input if input else params['input']
        self.output = output if output else params['output']

        self.name = 'f1'
        self.threshold = params['threshold']
        self.average = params['average']
        self.scored_birds = cfg['general']['scored_birds']
        self.reset()
    
    @torch.no_grad()
    def update(self, preds, data):
        '''
            :param: preds - tuple, predictions from the model
            :param: data - dict from dataset
        '''
        input = preds[self.input]
        target = data[self.output]
        input = input[:, self.scored_birds]
        target = target[:, self.scored_birds]
        
        class_pred = input >= self.threshold
        self.preds.extend(class_pred.detach().cpu().numpy())
        self.gt.extend(target.detach().cpu().numpy())
    
    def compute(self):
        self.gt = np.array(self.gt)
        self.preds = np.array(self.preds)
        return f1_score(self.gt, self.preds, average=self.average)

    def reset(self):
        self.preds = []
        self.gt = []

