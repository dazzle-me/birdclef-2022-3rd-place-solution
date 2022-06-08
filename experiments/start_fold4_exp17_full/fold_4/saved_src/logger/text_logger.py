from time import gmtime, strftime
from os.path import join
from torch.utils.tensorboard import SummaryWriter

class TextLogger():
    def __init__(self, cfg, exp_dir):
        ## On previous steps we assured that this file is new 
        ## and hence we won't overwrite any information
        self.file = join(exp_dir, cfg['logger']['output_file'])
        self.writer = SummaryWriter(
            log_dir=exp_dir,
        )

    def __call__(self, message):
        time_when_called = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        with open(self.file, 'a') as file:
            file.write(f'[{time_when_called}] : {message}')
            file.write('\n')
        print(f'[{time_when_called}] : {message}')
    
    def log_value(self, name, value, step):
        self.writer.add_scalar(name, value, step)