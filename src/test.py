import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from src.models.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
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

_parse_args()