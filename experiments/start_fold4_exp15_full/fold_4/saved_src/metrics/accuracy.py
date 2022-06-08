import torch

class Accuracy():
    def __init__(self, cfg, input=None, output=None):
        params = cfg['metrics']['accuracy']

        self.input = input if input else params['input']
        self.output = output if output else params['output']

        self.name = 'accuracy'
        self.reset()
    
    @torch.no_grad()
    def update(self, preds, data):
        '''
            :param: preds - tuple, predictions from the model
            :param: data - dict from dataset
        '''
        input = preds[self.input]
        target = data[self.output]

        input = torch.argmax(input, dim=1)
        assert len(input) == len(target), f"len(preds) : {len(input)}, len(target) : {len(target)}"
        self.good_samples += (input == target.squeeze()).float().sum()
        self.all_samples += len(input)
    
    def compute(self):
        return self.good_samples / self.all_samples

    def reset(self):
        self.good_samples = 0
        self.all_samples = 0

