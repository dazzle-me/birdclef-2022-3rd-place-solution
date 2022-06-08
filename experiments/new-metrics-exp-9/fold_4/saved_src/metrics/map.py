import torch

class mAP():
    def __init__(self, cfg, input=None, output=None):
        params = cfg['metrics']['mAP']
        
        self.input = input if input else params['input'] 
        self.output = output if output else params['output']
        
        self.at = params['k']
        self.precision_multiplier = torch.Tensor([1 / k for k in range(1, self.at + 1)]).to(cfg['general']['device'])
        
        self.name = 'mAP'
        self.reset()

    @torch.no_grad()
    def update(self, preds, data):
        '''
            :param: preds - tuple, predictions from the model
            :param: data - dict from dataset
        '''

        input = preds[self.input]
        target = data[self.output]
        
        ### target.shape == (batch_size, 1)
        ### preds are either similarity score or probability score (the same concept)
        _, indices = torch.sort(input, dim=1, descending=True)
        batch_size, _ = indices.shape
        
        # print(
        #     f"target.shape : {target.shape}", target
        # )
        ## create a view
        target = target.expand((batch_size, self.at))
        # print(
        #     f"target.shape : {target.shape}", target
        # )
        # print(
        #     f"precision_multiplier.shape : {self.precision_multiplier.shape}", self.precision_multiplier
        # )
        precision_multiplier = self.precision_multiplier.expand((batch_size, self.at))
        # print(
        #     f"precision_multiplier.shape : {precision_multiplier.shape}", precision_multiplier
        # )
        # print(
        #     f"preds.shape : {indices[:, :self.at].shape}\n\
        #       precision_multiplier.shape : {precision_multiplier.shape}\n\
        #       target.shape : {target.shape}"
        # )
        # print(
        #     f"indices : {indices}"
        # )
        # print(
        #     f"taget : {target}"
        # )
        # print(
        #     f"preds : {preds}"
        # )
        delta = (precision_multiplier * (target == indices[:, :self.at]).float()).sum()
        self.value += delta
        self.num_samples += len(target)
    
    def compute(self):
        return self.value / self.num_samples

    def reset(self):
        self.value = 0
        self.num_samples = 0