import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda
from warmupMultiStepLR import WarmupMultiStepLR

def train_step(step, optimizer, model, match_loss, data, scheduler,
               do_zero_grad=True, do_optim_step=True, accum_steps=1):
    model.train()
    xs = data['xs']
    ys = data['ys'].squeeze(-1)
    logits, ys_ds, e_hat, y_hat = model(xs, ys)
    loss, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, logits, ys_ds, e_hat, y_hat)
    loss_val = [geo_loss, cla_loss, l2_loss]
    # scale loss for gradient accumulation
    (loss / accum_steps).backward()
    if do_zero_grad:
        optimizer.zero_grad()
    if do_optim_step:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return loss_val


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    scheduler = None #config.scheduler
    scheduler = WarmupMultiStepLR(optimizer, [200000,400000], warmup_iters=100000, warmup_factor=0.01,warmup_method='linear')
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss']*(config.iter_num+1))
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])
    accum_steps = getattr(config, 'accum_steps', 1)
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        # Gradient accumulation: collect accum_steps mini-batches
        cur_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        loss_vals = None
        valid_accum = 0
        for accum_i in range(accum_steps):
            try:
                train_data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                train_data = next(train_loader_iter)
            train_data = tocuda(train_data)
            is_last = (accum_i == accum_steps - 1)
            try:
                vals = train_step(
                    step, optimizer, model, match_loss, train_data, scheduler,
                    do_zero_grad=False,
                    do_optim_step=is_last,
                    accum_steps=accum_steps
                )
                loss_vals = vals
                valid_accum += 1
            except Exception:
                continue
        if loss_vals is None:
            continue
        logger_train.append([cur_lr] + loss_vals)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  _, _, _  = valid(valid_loader, model, step, config)
            logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

