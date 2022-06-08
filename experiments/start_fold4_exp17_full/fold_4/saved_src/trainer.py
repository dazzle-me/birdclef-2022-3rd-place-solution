import os
from os.path import join
import gc

import json
import shutil
import pandas as pd

import time
from time import strftime

import torch
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from glob import glob

## custom packages
from logger.text_logger import TextLogger

from utils.average_meter import AverageMeter
from utils.sliding_window import AverageWindow

from utils.save_model import SaveBestHandler
from utils.early_stopping import EarlyStopping

from models.model_factory import ModelFactory

from optimization.optimizer import OptimizerFactory
from optimization.loss import LossFactory
from optimization.scheduler import SchedulerFactory

from dataset.dataset import CustomDataset
from dataset.sampler import BalancedSampler

from metrics.list_constructor import list_constructor

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss = LossFactory().get_loss(self.cfg, mode='train')
        self.validation_loss = LossFactory().get_loss(self.cfg, mode='val')
        self.metric_list = list_constructor(cfg)
        self.metric_dict = {}
        if self.cfg['training']['use_fp16']:
            self.scaler = GradScaler()

        self.average_time = AverageWindow(size=cfg['logger']['window_size'])

    def _save_source_files(self, exp_dir):
        for file in [item for item in glob('**/**', recursive=True) \
                     if 'experiments' not in item \
                     and ('.py' in item or '.yaml' in item) \
                     and '__' not in item]:
            destination = os.path.dirname(os.path.join(exp_dir, 'saved_src', file))
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.copy(file, destination)

    def _prepare_experiment_dir(self, fold):
        exp_root_dir = self.cfg['general']['exp_root']
        os.makedirs(exp_root_dir, exist_ok=True)

        exp_name = f"{self.cfg['general']['exp_name']}"
        exp_dir = os.path.join(exp_root_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=self.cfg['general']['dev'])
        fold_dir_path = os.path.join(exp_dir, f'fold_{fold}')
        self.cfg['exp_path'] = fold_dir_path
        self._save_source_files(fold_dir_path)

        self.exp_dir = fold_dir_path
        if self.cfg['logger']['name'].lower() == 'text_logger':
            self.logger = TextLogger(self.cfg, self.exp_dir)
        else:
            raise ValueError("Others loggers than 'TextLogger' are yet to be supported")
        os.makedirs(join(self.exp_dir, 'weights'), exist_ok=self.cfg['general']['dev'])
        self.save_handler = SaveBestHandler(
            join(self.exp_dir, 'weights'),
            self.logger,
            self.cfg['utils']['save_handler']['mode'],
            self.cfg['utils']['save_handler']['top_1'],
            self.cfg['utils']['save_handler']['save_all']
        )
        self.early_stopping = EarlyStopping(
            self.cfg['utils']['early_stopping']['patience'],
            self.cfg['utils']['early_stopping']['mode']
        )
        self._reinit_training()

    def _reinit_training(self):
        self.model = ModelFactory().get_model(self.cfg).to(self.cfg['general']['device'])
        self.model.is_train = True
        self.opt = OptimizerFactory().get_optim(self.cfg, self.model.parameters())
        self.scheduler = SchedulerFactory().get_scheduler(self.cfg, self.opt, train_steps=self.train_steps, warmup_steps=self.warmup_steps)
        for metric in self.metric_list:
            metric.reset()
        if self.cfg['general']['weights_path'] is not None:
            success = self.model.load_state_dict(torch.load(self.cfg['general']['weights_path']))
            if success:
                self.logger(f"Model weights load successfull, weights path : {self.cfg['general']['weights_path']}")
            else:
                self.logger(f"Failed to load model weights, weights path : {self.cfg['general']['weights_path']}")
        else:
            self.logger("No pretrained weights were specified for this run (manually)")

    def train(self):
        if self.cfg['training']['k_fold']:
            num_folds = self.cfg['training']['num_folds']
            for fold in range(num_folds):
                train_folds = [f for f in range(num_folds)]
                del train_folds[fold]
                val_folds = [fold]
                self.train_single_fold(train_folds, val_folds)
        else:
            try:
                train_folds = self.cfg['training']['train_folds']
                val_folds = self.cfg['training']['val_folds']
            except:
                ## dummy folds, to replace in future
                train_folds = [1, 2, 3, 4]
                val_folds = [0]
            self.train_single_fold(train_folds, val_folds)

    def train_single_fold(self, train_folds, val_folds):
        cfg = self.cfg
        train_ds = CustomDataset(cfg, train=True)
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg['data']['batch_size'],
            num_workers=cfg['data']['num_workers'],
            sampler=BalancedSampler(cfg) if cfg['data']['use_sampler'] else None,
            shuffle=False if cfg['data']['use_sampler'] else True,
            drop_last=False if cfg['data']['use_sampler'] else True,
            pin_memory=cfg['data']['pin_memory_train']
        )
        val_ds = CustomDataset(cfg, train=False)
        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg['data']['val_batch_size'],
            num_workers=cfg['data']['num_workers'],
            shuffle=False,
            drop_last=False,
            pin_memory=cfg['data']['pin_memory_val']
        )
        if self.cfg['training']['scheduler']['name'] == 'CosineLRScheduler':
            sched_params = self.cfg['training']['scheduler']['CosineLRScheduler']
            self.train_steps = len(train_dl) * sched_params['cycle_length']
            self.warmup_steps = sched_params['warmup_epochs'] * len(train_dl)
        else:
            self.train_steps = None
            self.warmup_steps = None

        fold = val_folds[0]
        self._prepare_experiment_dir(fold)
        with open(join(self.exp_dir, 'birds_meta_array.json'), 'w') as output_file:
            json.dump(train_ds.bird2id_array, output_file, indent=4)
        with open(join(self.exp_dir, 'birds_meta.json'), 'w') as output_file:
            json.dump(train_ds.bird2id, output_file, indent=4)

        self.logger(f"Starting fold-{fold} fold training")
        self.logger(f"Training folds : {train_folds}")
        self.logger(f"Val folds : {val_folds}")
        self.logger(f"Train length : {len(train_dl)}, val length : {len(val_dl)}")

        if self.cfg['general']['dev']:
            metric = self.validate(val_dl, 0)
        for epoch in range(1, self.cfg['training']['epochs'] + 1):
            gc.collect()
            self.train_one_epoch(train_dl, epoch)
            metric = self.validate(val_dl, epoch)
            self.save_handler(self.model, epoch, metric, optimizer=self.opt, scheduler=self.scheduler)
            if cfg['training']['scheduler']['name'] == 'LambdaLR':
                self.scheduler.step()
            elif cfg['training']['scheduler']['name'] == 'ReduceLROnPlateau':
                self.scheduler.step(metric)
            interrupt_training = self.early_stopping(metric)
            if interrupt_training:
                self.logger(f"Training stopped due to early stopping, since metric won't improve for the last {self.early_stopping.patience} epochs")
                break

    def train_one_epoch(self, train_loader, epoch):
        avg_meter = AverageMeter()
        for batch_idx, data in enumerate(train_loader):
            start_time = time.time()
            data = self.move_to_device(
                data,
                self.cfg['general']['fields_to_move'],
                self.cfg['general']['device']
            )
            self.opt.zero_grad()

            if self.cfg['training']['use_fp16']:
                with autocast():
                    preds = self.model.train_step(data)
                    loss = self.loss(preds, data)
            else:
                preds = self.model.train_step(data)
                loss = self.loss(preds, data)

            avg_meter.update(loss.item(), 1)
            if self.cfg['training']['use_fp16']:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.cfg['training']['clip_grad_norm']:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.gradient_clipping_norm
                )
            else:
                grad_norm = 'not_used'

            if self.cfg['training']['use_fp16']:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            if self.cfg['training']['scheduler']['name'] == 'CosineLRScheduler':
                self.scheduler.step_update((epoch - 1) * len(train_loader) + batch_idx)

            for metric in self.metric_list:
                metric.update(preds, data)

            if (batch_idx + 1) % self.cfg['logger']['frequency'] == 0:
                self.logger(f"Epoch - [{epoch:03d}/{self.cfg['training']['epochs']}], step - [{batch_idx:03d}/{len(train_loader):03d}], current loss = {loss.item():.5f}, lr = {self.opt.param_groups[0]['lr']:.6f}")
                if self.cfg['training']['clip_grad_norm']:
                    self.logger("grad norm : {grad_norm:.4f}")

                remaining_time = self.average_time.compute() * ((len(train_loader) - (batch_idx + 1)) + (self.cfg['training']['epochs'] - epoch) * len(train_loader))
                remaining_time = int(remaining_time)
                m, s = divmod(remaining_time, 60)
                h, m = divmod(m, 60)
                d, h = divmod(h, 60)
                self.logger(f"ETA - {d:01d} days, {h:02d}:{m:02d}:{s:02d}")

            self.logger.log_value('lr', self.opt.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
            end_time = time.time()
            self.average_time.update(end_time - start_time)

        for metric in self.metric_list:
            value = metric.compute()
            self.logger.log_value(f'{metric.name}/train', value, epoch)
            self.logger(f'{metric.name}/train - {value:.4f}')
            self.metric_dict[metric.name] = value
            metric.reset()

        self.logger.log_value('loss/train', avg_meter.average, epoch)
        self.logger(f"Epoch : [{epoch:03d}/{self.cfg['training']['epochs']}], Train loss : {avg_meter.average:.4f}, lr : {self.opt.param_groups[0]['lr']:.6f}")
        self.logger('-' * 50)

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        self.model.is_train = False
        average_meter = AverageMeter()

        if self.cfg['training']['save_val']:
            pred_arr = []
            targets_arr = []

        for batch_idx, data in enumerate(val_loader):
            data = self.move_to_device(
                data,
                self.cfg['general']['fields_to_move'],
                self.cfg['general']['device']
            )
            preds = self.model.train_step(data)

            if self.cfg['training']['save_val']:
                pred_arr.extend(preds['logits'].detach().cpu().numpy())
                targets_arr.extend(data['target'].detach().cpu().numpy())

            loss = self.validation_loss(preds, data)
            for metric in self.metric_list:
                metric.update(preds, data)


            ## account for non-equal-sized last batch with this hack
            average_meter.update(loss.item(), len(data[self.cfg['general']['target_field']]) / self.cfg['data']['val_batch_size'])
            ## single usage of :param: target_field
            ############################
        for metric in self.metric_list:
            value = metric.compute()
            self.logger.log_value(f'{metric.name}/val', value, epoch)
            self.logger(f'{metric.name}/val - {value:.4f}')
            self.metric_dict[metric.name] = value
            metric.reset()
        validation_loss = average_meter.average
        self.metric_dict['validation_loss'] = validation_loss
        self.logger.log_value(f'loss/val', validation_loss, epoch)
        self.logger(f"Epoch : [{epoch:03d}/{self.cfg['training']['epochs']}], Validation loss : {validation_loss:.4f}")
        self.logger('-' * 50)
        self.model.train()
        self.model.is_train = True
        if self.cfg['training']['save_val']:
            print(np.stack(pred_arr).shape, np.stack(targets_arr).shape)
            data = {
                'metric_dict' : self.metric_dict,
                'predictions' : np.stack(pred_arr).tolist(),
                'targets' : np.stack(targets_arr).tolist()
            }
            os.makedirs(join(self.exp_dir, 'val_output'), exist_ok=True)
            with open(join(self.exp_dir, f'val_output/epoch_{epoch}_validation.json'), 'w') as output_file:
                json.dump(data, output_file, indent=4)
        return self.metric_dict[self.cfg['utils']['save_handler']['monitor']]

    @staticmethod
    def move_to_device(data, fields, device):
        for field in fields:
            data[field] = data[field].to(device)
        return data

    def _load_best_model(self):
        return ModelFactory().get_model(self.cfg, weights_path=self.save_handler.get_best_model_path())
