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
       'block_channels': 8,
       'residual_chanels': 16,
       'skip_channels': 32,
       'end_channels': 6,
       'categories': 4,
       'device': 'cpu'}

net = WaveNet(**arg)

input = [i % 4 for i in range(net.receptive_field)]
input1 = net.one_hot([input])
input2 = net.one_hot([input, input])

#print(net(input1))
#print(net(input1, probs=True))

def test_WaveBlock():
    block = WaveBlock(residual_channels=arg['residual_chanels'],
                      block_channels=arg['block_channels'],
                      skip_channels=arg['skip_channels'],
                      kernel_size=arg['kernel_size'],
                      dilation_rate=2)

    input = [i % arg['residual_chanels'] for i in range(net.receptive_field)]
    input1 = torch.nn.functional.one_hot(torch.as_tensor([input]), arg['residual_chanels']).type(torch.FloatTensor).permute(0,2,1)
    input2 = torch.nn.functional.one_hot(torch.as_tensor([input, input]), arg['residual_chanels']).type(torch.FloatTensor).permute(0,2,1)

    resid, skip = block(input1)
    assert resid.shape == (1, arg['residual_chanels'], input1.shape[-1] - 2)
    assert skip.shape == (1, arg['skip_channels'], input1.shape[-1] - 2)

    resid, skip = block(input2)
    assert resid.shape == (2, arg['residual_chanels'], input1.shape[-1] - 2)
    assert skip.shape == (2, arg['skip_channels'], input1.shape[-1] - 2)



def test_WaveNet_one_hot():
    assert input1.shape == (1, arg['categories'], net.receptive_field)
    assert input2.shape == (2, arg['categories'], net.receptive_field)


def test_WaveNet_forward():
    assert net(input1).shape == (1, arg['categories'], 1)
    assert net(input2).shape == (2, arg['categories'], 1)
    assert net(torch.cat((input2, input2))).shape == (2, arg['categories'], input2.shape[-2])
    assert torch.sum(net(input1, probs=True), dim=1) == 1



