import sys, os
import torch
from torch import nn, optim


# import project root to PYTHONPATH
abs_dir = os.path.dirname(os.path.abspath(__file__)) # abs adress of this file
root_folder = abs_dir.split('tests')[0]
sys.path.append(root_folder)

from models.wave_net_pytorch import WaveNet, WaveBlock, WaveGenerator

arg = {'dilations': [2 ** i for j in range(5) for i in range(10)],
       'kernel_size': 2,
       'block_channels': 4,
       'residual_chanels': 4,
       'skip_channels': 4,
       'end_channels': 4,
       'categories': 4,
       'device': 'cpu'}

net = WaveNet(**arg)

input = [i % 4 for i in range(net.receptive_field)]
input1 = net.one_hot([input])
input2 = net.one_hot([input, input])


def test_WaveBlock():
    block = WaveBlock(residual_channels=arg['residual_chanels'],
                      block_channels=arg['block_channels'],
                      skip_channels=arg['skip_channels'],
                      kernel_size=arg['kernel_size'],
                      dilation_rate=2)

    assert block(input1).shape == (1, arg['categories'], input1.shape[-1] - 2)


def test_WaveNet_one_hot():
    assert input1.shape == (1, arg['categories'], net.receptive_field)
    assert input2.shape == (2, arg['categories'], net.receptive_field)


def test_WaveNet_forward():
    assert net(input1).shape == (1, arg['categories'], 1)
    assert net(input2).shape == (2, arg['categories'], 1)


