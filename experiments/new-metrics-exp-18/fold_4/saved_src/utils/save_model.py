import os
import numpy as np
import torch

class SaveBestHandler():
    def __init__(self, save_path, logger, mode='min', top_1=False, save_all=False):
        self.root = save_path
        self.best_model_name_file = os.path.join(save_path, 'best_model.txt')
        self.mode = mode
        self.logger = logger
        self.top_1 = top_1
        self.save_all = save_all
        if mode == 'min':
            self.best_metric = np.inf
        elif mode == 'max':
            self.best_metric = -np.inf
        else:
            raise ValueError(f"Mode : {mode} isn't supported")

    def get_best_model_name(self):
        with open(self.best_model_name_file, 'r') as file:
            best_model_name = file.readline()
        return best_model_name

    def get_best_model_path(self):
        return os.path.join(self.root, self.get_best_model_name())

    # def __call__(self, current_model, epoch, metric):
    #     if self.mode == 'min':
    #         condition = metric < self.best_metric
    #     elif self.mode == 'max':
    #         condition = metric > self.best_metric
    #     if condition:
    #         self.logger(f"Saving mode on epoch {epoch}, previous best metric : {self.best_metric:.4f} -> new best metric : {metric:.4f}")
    #         ## Try to delete checkpoint only if we saved the model
    #         ## at least one time
    #         if self.best_metric not in [np.inf, -np.inf] and self.top_1:
    #             previous_model_path = self.get_best_model_path()
    #             if os.path.exists(previous_model_path):
    #                 os.remove(previous_model_path)

    #         new_model_name = f'epoch_{epoch}_metric_{metric:.4f}.pth'
    #         with open(self.best_model_name_file, 'w') as file:
    #             file.write(new_model_name)
    #         torch.save(current_model.state_dict(), os.path.join(self.root, new_model_name))
    #         self.best_metric = metric

    def __call__(self, current_model, epoch, metric, optimizer=None, scheduler=None):
        if self.mode == 'min':
            condition = metric < self.best_metric
        elif self.mode == 'max':
            condition = metric > self.best_metric
        if condition or self.save_all:
            self.logger(f"Saving model on epoch {epoch}, previous best metric : {self.best_metric:.4f} -> new best metric : {metric:.4f}")
            ## Try to delete checkpoint only if we saved the model
            ## at least one time
            if self.best_metric not in [np.inf, -np.inf] and self.top_1:
                previous_model_path = self.get_best_model_path()
                if os.path.exists(previous_model_path):
                    os.remove(previous_model_path)

            new_model_name = f'epoch_{epoch}_metric_{metric:.4f}.pth'
            with open(self.best_model_name_file, 'w') as file:
                file.write(new_model_name)
            checkpoint = {
                'epoch' : epoch,
                'model' : current_model.state_dict()
            }
            if optimizer is not None:
                checkpoint['optimizer'] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, os.path.join(self.root, new_model_name))
            self.best_metric = metric
