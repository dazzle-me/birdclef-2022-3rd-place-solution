import numpy as np

class EarlyStopping():
    def __init__(self, patience, mode):
        '''
            :param: patience - positive int
            :param: mode - ['min', 'max']

            if mode == 'min', earlystopping handler will
            stop the training if the monitored metric won't go
            down for :param: patience epochs,

            for mode == 'max', however
            the training will be stopped if the metric won't go
            up for :param: patience epochs.
        '''
        self.patience = patience
        self.current_wait = 0
        self.mode = mode
        if mode == 'min':
            self.best_metric = np.inf  
        elif mode == 'max':
            self.best_metric = -np.inf
        else:
            raise ValueError(f"Mode : {mode} isn't supported")
            
    def __call__(self, metric):
        if self.mode == 'min':
            condition = metric < self.best_metric
        elif self.mode == 'max':
            condition = metric > self.best_metric
        if condition:
            self.current_wait = 0
            self.best_metric = metric
        else:
            self.current_wait += 1
        interrupt_training = self.current_wait > self.patience
        return interrupt_training