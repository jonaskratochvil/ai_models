import sys, os
import torch


# import project root to PYTHONPATH
abs_dir = os.path.dirname(os.path.abspath(__file__)) # abs adress of this file
root_folder = abs_dir.split('tests')[0]
sys.path.append(root_folder)

from models.wave_net_pytorch import WaveNet, WaveBlock


# TODO: write test class for WaveNet

dilations = [2**i for j in range(5) for i in range(5)]

net = WaveNet(filters=15,
              kernel_size=2,
              dilations=dilations,
              categories=3,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

block = WaveBlock(filters=3, kernel_size=2, dilation_rate=2)

batch = torch.Tensor([[[0, 0, 0, 1],
                       [1, 0, 1, 0],
                       [0, 1, 0, 0]],
                      [[1, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]]])

print(block.forward(batch)[0].shape)
