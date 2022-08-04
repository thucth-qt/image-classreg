import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from src import utils
from src.models.models import create_model, safe_model_name, resume_checkpoint, model_parameters
from src.datasets.load_data import *
from src.datasets import preprocess
from src.metrics.losses import *
from src.optim import create_optimizer_v2, optimizer_kwargs
from src.scheduler import create_scheduler
from src.utils import get_parser

_logger = logging.getLogger('train')


def main():
    utils.setup_default_logging()
    args, args_text = get_parser()
    _logger.info('Training start...')
    utils.random_seed(1101)

    model = create_model(
        args.model,
        pretrained=args.pretrained, #load pretrained weights from the internet
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate)
        
    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

        # setup loss function
    train_loss_fn = globals()[args.loss].cuda()
    validate_loss_fn = globals()[args.loss].cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    
    # Dataset

    # create the train and eval datasets
    tg_tf = preprocess.create_target_tf()
    train_tf = preprocess.create_train_tf(input_size=args.input_size[:-1], means=args.mean, stds=args.std)
    val_tf = preprocess.create_val_tf(input_size=args.input_size[:-1], means=args.mean, stds=args.std)

    train_ds = DatasetFromCsv(annotations_file=args.train_ds,
                                transform=train_tf,
                                target_transform=tg_tf)

    val_ds = DatasetFromCsv(annotations_file=args.val_ds,
                            transform=val_tf,
                            target_transform=tg_tf)

    train_loader = create_dataloader(train_ds, \
        batch_size=args.batch_size, \
        workers=args.workers, \
        shuffle=True)
    val_loader = create_dataloader(val_ds, \
        batch_size=args.validation_batch_size, \
        workers=args.workers, \
        shuffle=True)

    # IO training
    if args.experiment:
            exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str("_".join([str(x) for x in args.input_size[:-1]]))
        ])
    output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    saver = utils.CheckpointSaver(
        model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=10)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(
                epoch, model, train_loader, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver)

            eval_metrics = validate(model, val_loader, validate_loss_fn, args, amp_autocast=amp_autocast)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, amp_autocast=suppress):

    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.cuda(), target.cuda()
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx %args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            losses_m.update(loss.data.item(), input.size(0))

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

        if last_batch or (batch_idx + 1) % 10 == 0:
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch or batch_idx % args.log_interval == 0:
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
