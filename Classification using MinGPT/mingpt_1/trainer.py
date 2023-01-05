"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from mingpt_1.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, validation_dataset, downstream_finetune=False, collate_fn=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.callbacks = defaultdict(list)
        self.collate_fn = collate_fn

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.train_acc = 0.0
        self.val_acc = 0.0
        self.downstream_finetune=downstream_finetune

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=lambda batch: self.collate_fn(batch, self.device)
        )

        val_loader = DataLoader(
            self.validation_dataset,
            sampler=torch.utils.data.RandomSampler(self.validation_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=lambda batch: self.collate_fn(batch, self.device)
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        self.train_acc = 0.0
        self.val_acc = 0.0
        data_iter = iter(train_loader)
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        iter_list = []
        for i in range(config.max_iters):
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y, self.downstream_finetune)

            logits_n = torch.sigmoid(logits.cpu())
            logit = []
            for i in logits_n:
                logit.append(torch.argmax(i))
            logit = torch.tensor(logit).unsqueeze(1).to(self.device)
            acc = (logit == y).float().sum()
            self.train_acc += (acc / config.batch_size)
            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            model.eval()
            with torch.no_grad():
                val_iter = iter(val_loader)
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)
                val_batch = [t.to(self.device) for t in val_batch]
                x, y = val_batch
                logits, self.val_loss = model(x, y, self.downstream_finetune)
                logits_n = torch.sigmoid(logits.cpu())
                logit = []
                for i in logits_n:
                    logit.append(torch.argmax(i))
                logit = torch.tensor(logit).unsqueeze(1).to(self.device)
                acc = (logit == y).float().sum()
                self.val_acc += (acc / config.batch_size)

            if self.iter_num % 100 == 0:
                self.train_acc = self.train_acc / 100
                self.val_acc = self.val_acc / 100
                train_loss_list.append(self.loss.cpu().detach().numpy())
                train_acc_list.append(self.train_acc.cpu().detach().numpy())
                val_loss_list.append(self.val_loss.cpu().detach().numpy())
                val_acc_list.append(self.val_acc.cpu().detach().numpy())
                iter_list.append(self.iter_num)
                self.trigger_callbacks('on_batch_end')
                if self.iter_num != 0:
                    self.train_acc = 0.0
                    self.val_acc = 0.0
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

        return iter_list, train_acc_list, train_loss_list, val_acc_list, val_loss_list
            # termination conditions
            # if config.max_iters is not None and self.iter_num >= config.max_iters:
            #     break
