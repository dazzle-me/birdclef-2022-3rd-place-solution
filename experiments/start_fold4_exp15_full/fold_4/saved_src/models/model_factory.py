import torch
from models.model import Net, RCNN

class ModelFactory():
    def __init__(self):
        self.supported_models = {
            'Net' : Net,
            'RCNN' : RCNN
        }
    def get_model(self, cfg, weights_path=None):
        if cfg['model']['name'] in self.supported_models.keys():
            model = self.supported_models[cfg['model']['name']](cfg)
            if weights_path is not None:
                model.load_state_dict(torch.load(weights_path))
            return model
        else:
            raise ValueError(f"Requested model : {cfg['model']['name']} is not supported")