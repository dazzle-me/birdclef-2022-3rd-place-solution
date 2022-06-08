from sched import scheduler
import torch

from optimization.cosine_scheduler import CosineLRScheduler, Scheduler

class SchedulerFactory():
    def __init__(self):
        self.supported_schedulers = {
            'LambdaLR' : torch.optim.lr_scheduler.LambdaLR,
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CosineLRScheduler' : CosineLRScheduler
        }
    @staticmethod
    def lrfn(epoch, 
             lr_start = 0.0001, 
             lr_max = 0.000015, 
             lr_min = 0.0000001, 
             lr_ramp_ep = 1, 
             lr_sus_ep  = 0, 
             lr_decay   = 0.95):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start   
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max    
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min 
        return lr
    
    def get_scheduler(self, cfg, optim, train_steps=None, warmup_steps=None):
        params = cfg['training']['scheduler']

        if params['name'] in self.supported_schedulers.keys():
            scheduler_params = params[params['name']]
            if params['name'] == 'LambdaLR':
                return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch : \
                                self.lrfn(epoch, lr_start=cfg.lr_start, lr_max=cfg.lr_per_sample * cfg.train_batchsize, lr_min=cfg.lr_min))
            elif params['name'] == 'ReduceLROnPlateau':
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim,
                    mode=scheduler_params['mode'],
                    factor=scheduler_params['factor'],
                    patience=scheduler_params['patience'],
                    min_lr=scheduler_params['min_lr'],
                    threshold=scheduler_params['delta'],
                )
            elif params['name'] == 'CosineLRScheduler':
                return CosineLRScheduler(
                    optim,
                    t_initial=train_steps,
                    cycle_mul=1.,
                    lr_min=scheduler_params['lr_min'],
                    warmup_lr_init=scheduler_params['lr_warmup'],
                    cycle_decay=scheduler_params['cycle_decay'],
                    cycle_limit=scheduler_params['cycle_limit'],
                    warmup_t=warmup_steps,
                    t_in_epochs=False,
                )
        else:
            raise ValueError(f"Optimizer : {params['name']} isn't supported")
