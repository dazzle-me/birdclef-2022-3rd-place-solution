import sklearn
import torch
from sklearn.metrics import f1_score
import numpy as np

from pprint import pprint

class F1():
    def __init__(self, cfg, params, input=None, output=None):
        self.input = input if input else params['input']
        self.output = output if output else params['output']

        self.name = params['name']
        self.threshold = params['threshold']
        self.average = params['average']

        scored_birds = cfg['general']['scored_birds']
        scored_birds_names = cfg['general']['scored_birds_names']

        self.score_only_birds = params['score_birds']
        
        if self.score_only_birds != 'all':
            self.scored_birds = []
            self.scored_birds_names = []
            for bird in self.score_only_birds:
                idx = scored_birds_names.index(bird)
                self.scored_birds.append(scored_birds[idx])
                self.scored_birds_names.append(bird)
        else:
            self.scored_birds = scored_birds
            self.scored_birds_names = scored_birds_names
        print(self.scored_birds, '\n', self.scored_birds_names)
        self.mapping = {k:v for k, v in zip(self.scored_birds_names, self.scored_birds)}
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
        if self.average in ['micro', 'macro', 'samples']:
            return f1_score(self.gt, self.preds, average=self.average)
        else: ## average None
            names = [f"{self.name}_{bird}" for bird in self.scored_birds_names]
            return {k:v for k, v in zip(names, f1_score(self.gt, self.preds, average=None))}

    def reset(self):
        self.preds = []
        self.gt = []

