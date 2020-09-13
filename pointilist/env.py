#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import pathlib
import functools

import torch
import torchvision


DATA_CACHE = pathlib.Path(__file__).parents[1] / 'data'
# a convenience for the MNIST dataset
# set `train` and transforms as per
# MNIST = functools.partial(torchvision.datasets.MNIST, DATA_CACHE, download=True)

# convenience var to move to cpu/cuda
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')