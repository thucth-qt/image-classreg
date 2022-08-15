import argparse
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
from src.datasets.load_data import train_ds, val_ds, create_dataloader
from src.metrics.losses import *
from src.optim import create_optimizer_v2, optimizer_kwargs
from src.scheduler import create_scheduler

_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='Image tasks')
parser.add_argument('-c', '--config', default='src/strategies/efficientnet_reg.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
args = parser.parse_args()


def _parse_args():
    # Do we have a config file to parse?
    global args
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    print(args)
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    print(args)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    _logger.info('Training start...')
    utils.random_seed(1101)

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)
    model = create_model(
        "efficientnet_b0",
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))
