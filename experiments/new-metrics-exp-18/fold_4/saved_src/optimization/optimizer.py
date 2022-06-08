import torch

## Can be done w/o class?
class OptimizerFactory():
    def __init__(self):
        self.supported_optimizers = {
            'Adam' : torch.optim.Adam,
            'SGD' : torch.optim.SGD,
            'AdamW' : torch.optim.AdamW
        }
    def get_optim(self, cfg, model_params):
        params = cfg['training']['optimizer']
        if params['name'] in self.supported_optimizers.keys():
            if params['name'] == 'Adam':
                return torch.optim.Adam(model_params, lr=params['lr'], weight_decay=params['wd'])
            elif params['name'] == 'SGD':
                return torch.optim.SGD(model_params, lr=params['lr'], momentum=params['momentum'], weight_decay=params['wd'])
            elif params['name'] == 'AdamW':
                return torch.optim.AdamW(model_params, lr=params['lr'], weight_decay=params['wd'])
        else:
            raise ValueError(f"Optimizer : {params['name']} isn't supported")
        