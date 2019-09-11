import sys, os
import torch
from torch import nn, optim


# import project root to PYTHONPATH
abs_dir = os.path.dirname(os.path.abspath(__file__)) # abs adress of this file
root_folder = abs_dir.split('tests')[0]
sys.path.append(root_folder)

from models.wave_net_pytorch import WaveNet, WaveBlock, WaveGenerator


# TODO: write test class for WaveNet

dilations = [2**i for j in range(5) for i in range(5)]

net = WaveNet(filters=3,
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
                       [0, 0, 1, 0]]]).type(torch.FloatTensor)
target = torch.Tensor([[1, 2, 1, 0], [0, 0, 2, 1]]).type(torch.LongTensor)

#print(batch.type())
#print(net.one_hot([1]).unsqueeze(0).shape)
#print(net(batch[:1], probs=True)[0, :,-1])
#print(torch.multinomial(net(batch, probs=True)[0, :,-1], 1))
#gen = WaveGenerator(net, [1, 2, 1, 0])

#net.generate([1,2,1,0], 1000)
#for i in range(10000):
#   print('{}: {}'.format(i, next(gen)))

criterion = nn.CrossEntropyLoss()  # only accepts logits!
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for i in range(1000):
    inputs, targets = batch[:, :, :-1], target[:, 1:]

    optimizer.zero_grad()  # zero the parameter gradients
    outputs = net(inputs, probs=False)  # output logits
    loss = criterion(outputs, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.step()
    print(loss.item())
