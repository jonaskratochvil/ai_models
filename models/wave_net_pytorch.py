import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import torch
from torch.utils.data.dataloader import DataLoader

from utils.utils import TensorQueue, Progbar


# jak jsou organizovane batche v torch?
# N - batch size
# D_in - input dimension
# H - hidden dimension
# D_out - output dimension
# C - number of channels
# L  - length of signal sequence

class WaveBlock(nn.Module):
    """One WaveNet stack

    #            |----------------------------------------|     *residual*
    #            |                                        |
    #            |    |-- conv -- tanh --|                |
    # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
    #                 |-- conv -- sigm --|     |
    #                                         1x1
    #                                          |
    # ---------------------------------------> + ------------->	*skip*

    ASCII art taken from https://github.com/vincentherrmann/pytorch-wavenet/blob/master/wavenet_model.py
    """
    def __init__(self, residual_channels, block_channels, skip_channels, kernel_size, dilation_rate):
        """
        :param residual_channels: Num. of channels for resid. connections between wave blocks
        :param block_channels: Num. of channels used inside wave blocks
        :param skip_channels: Num. of channels for skip connections directed to output
        :param kernel_size: Num. of branches for each convolution kernel
        :param dilation_rate: Hom much to dilate inputs before applying gate and filter
        """
        super(WaveBlock, self).__init__()

        self.filter = nn.Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.gate = nn.Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.conv1x1_resid = nn.Conv1d(block_channels, residual_channels, 1)
        self.conv1x1_skip = nn.Conv1d(block_channels, skip_channels, 1)

    def forward(self, residual):
        """Forward residual

        Convolution runs from left to right.
        Computed residual will be shorter than input residual due to dilation.
        We add only the overlapping part of input and output residuals.

        :param residual: Residual from previous block or from input_conv, (batch_size, channels, time_dim)
        :return: residual, skip
        """

        filter = torch.tanh(self.filter(residual))
        gate = torch.sigmoid(self.gate(residual))
        out = filter * gate
        residual = self.conv1x1_resid(out) + residual[..., -out.shape[-1]:]
        skip = self.conv1x1_skip(out)
        return residual, skip


class WaveNet(nn.Module):
    """


    """

    def __init__(self,
                 dilations: list,
                 kernel_size=2,
                 block_channels=32,
                 residual_chanels=32,
                 skip_channels=256,
                 end_channels=256,
                 categories=256,
                 device='cpu'):
        """
        :param dilations: list of dilations from first WaveBlock to last
        :param kernel_size: Num. of branches for each convolution kernel
        :param block_channels: Num. of channels used inside wave blocks
        :param residual_channels: Num. of channels for resid. connections between wave blocks
        :param skip_channels: Num. of channels for skip connections directed to output
        :param end_channels: Num. of channels for final 1x1 convolutions
        :param categories: Num. of predicted categories (ie sound bins)
        :param device: cpu or cuda
        """
        super(WaveNet, self).__init__()

        self.input_conv = nn.Conv1d(categories, residual_chanels, 1, 1)
        self.blocks = nn.ModuleList([WaveBlock(residual_chanels, block_channels, skip_channels, kernel_size,d)
                                     for d in dilations])

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.conv2 = nn.Conv1d(end_channels, categories, 1)
        self.softmax = nn.Softmax(dim=1)

        # params
        self.kernel_size = kernel_size
        self.block_channels = block_channels
        self.residual_chanels = residual_chanels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.categories = categories
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.categories = categories
        self.device = device
        self.receptive_field = sum(self.dilations) + 1

        # TODO: parametrize by user
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.to(device)
        print('Sending network to ', device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def logits_from_skip(self, skip):
        """Final layer

        Accept sum of all skip connections and run final 1x1 convs

        :param skip: sum of skip connections from WaveBlocks
        :return: model logits
        """
        skip = self.activation(skip)
        skip = self.conv1(skip)
        skip = self.activation(skip)
        skip = self.conv2(skip)
        return skip

    def forward(self, x: torch.Tensor, probs=False, temperature=1, capture_residuals=False):
        """Forward batch `x` through the network

        :param x: (batch_size, classes, time_dim)
        :param probs: If True, output softmax of logits
        :param temperature: Scale logits before softmax
        :param capture_residuals: If True, return list of residuals from all WaveBlocks too
        :return: output or (output, list_of_residuals)
        """

        if x.shape[-1] < self.receptive_field:
            raise Exception('Expected x.shape[-1]>={} but got x.shape[-1]={}'.format(self.receptive_field, x.shape[-1]))

        x = torch.as_tensor(x)
        x.to(self.device) # send input to device
        resid = self.input_conv(x)
        resids = [resid] if capture_residuals else None

        skips = 0
        for block in self.blocks:
            resid, skip = block(resid)
            skips = skip + skips[:, :, -skip.shape[-1]:] if \
                isinstance(skips, torch.Tensor) else skip
            if capture_residuals:
                resids.append(resid)

        logits = self.logits_from_skip(skips)

        if not self.training and temperature != 1:
            logits = logits / temperature

        if probs:
            out = self.softmax(logits)

        if capture_residuals:
            return out, resids
        else:
            return out

    def train_net(self, dataset: torch.utils.data.Dataset, epochs: int):
        """Backpropagation for given set of epochs

        TODO:
        1. figure out how to make batching more elegant
        2. Parametrize num_workers
        3. Integrate tensorboard logging

        There is some mess regarding batches:
        1. batch_size in dataset refers to how many segments to export from one audio.
        2. batch_size in DataLoader is set to None - leave the batching up to the dataset.

        :param dataset: A dataset object
        :param epochs: int
        :return:
        """
        criterion = nn.CrossEntropyLoss()  # only accepts logits!

        for epoch in range(epochs):
            data_loader = iter(DataLoader(dataset, batch_size=None, num_workers=0))
            running_loss = 0.0
            prgbar = Progbar(len(data_loader))
            for i, data in enumerate(data_loader):
                inputs, targets = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()  # zero the parameter gradients
                outputs = net(inputs, probs=False)  # output logits
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                prgbar.update(i)
            print(running_loss / len(data_loader))

    def one_hot(self, x: list or int):
        """One hot encode x, cast to FloatTensor

        :param x: integer, list, or list of lists
        :return: 3D tensor of shape (batch_size, channels, time)
        """

        if not isinstance(x, list):
            x = [x]

        if not isinstance(x[0], list):
            x = [x]
        elif isinstance(x[0][0], list):
            raise ValueError('Exceeded max list nesting. x must be a list or a list of lists.')

        return F.one_hot(torch.as_tensor(x), self.categories).type(torch.FloatTensor).permute(0,2,1).to(self.device)

    def time_concat(self, x1, x2):
        """Concatenate `x1` and `x2` in time dimension"""
        return torch.cat((x1, x2), 2)

    def generate(self, timesteps, x: list = None, temperature=1):
        """Slowly generate new samples

        Returns a generator object.
        If x is None, start is initialised with zeroes
        If x is shorter than receptive field, it is zero-padded

        TODO: DO not limit generation by timesteps

        :param timesteps: how many steps will be generated
        :param x: None or list of initial steps (eg a start of some music piece)
        :param temperature: Logit temperature for softmax
        :return: next_item, distrib
        """
        if x is None:
            input = torch.zeros(1, self.categories, self.receptive_field)
        else:
            pad = max(self.receptive_field - len(x), 0)
            if pad > 0:
                input = nn.ZeroPad2d((pad, 0, 0, 0))(self.one_hot([x]))
            else:
                input = self.one_hot([x])

        for i in range(timesteps):
            distrib = self.forward(input, probs=True, temperature=temperature)[0, :, -1]
            x = torch.multinomial(distrib, 1)[0]
            input = self.time_concat(input[:, :, 1:],
                                     self.one_hot([x]))
            # yield also distribution for comparison with other generation methods
            yield x.item(), distrib


class WaveGenerator:
    """Generator based on dynamic caching of previously calculated residuals.
    For more info see: https://github.com/tomlepaine/fast-wavenet
    """

    def __init__(self, wave_net: WaveNet, x: list = None, device = None):

        self.net = wave_net
        if device is not None:
            self.net.to_device(device)

        if x is None:
            input = torch.zeros(1, self.net.categories, self.net.receptive_field)
        else:
            pad = max(self.net.receptive_field - len(x), 0)
            if pad > 0:
                input = nn.ZeroPad2d((pad, 0, 0, 0))(self.net.one_hot([x]))
            else:
                input = self.net.one_hot([x])

        input = input.to(self.net.device)
        self.inputs_queue = TensorQueue(2, device=self.net.device)
        self.inputs_queue.push(input[..., -1:])

        with torch.set_grad_enabled(False):
            dist, residuals = self.net.forward(input, probs=True, capture_residuals=True)
        self.dist = dist[0, :, -1]
        self.out = self._sample_next(self.dist)

        self.queues = []
        for r, d in zip(residuals, self.net.dilations):
            self.queues.append(TensorQueue(d+1, self.net.device))
            self.queues[-1].push(r)

        self.first_run = True

    def __iter__(self):
        return self

    def _sample_next(self, distrib: torch.Tensor) -> torch.Tensor:
        """Sample class from distrib

        :param distrib: list or 1D tensor of integers
        :return: A 1D tensor
        """
        return torch.multinomial(distrib, 1)

    def __next__(self):
        """
        1. forward
        2. generate next input from output distribution
        3. store output as next input
        4. return output
        :return:
        """

        if self.first_run:
            self.first_run = False
            return self.out.item(), self.dist

        out = 0
        self.out = self.net.one_hot(self.out)
        self.inputs_queue.push(self.out)

        with torch.set_grad_enabled(False):
            residual = self.net.input_conv(self.inputs_queue.queue)
            for q, block in zip(self.queues, self.net.blocks):
                q.push(residual)
                residual, skip = block(q.queue, pad=False)
                out += skip  # should be output for a single time step
            out = self.net.logits_from_skip(out)
            distrib = self.net.softmax(out)[0, :, 0]  # flatten

        self.out = self._sample_next(distrib)
        return self.out.item(), distrib


if __name__ == '__main__':
    from data.audio.datasets_interface.PianoDataset import PianoDataset
    from data.audio.datasets_interface.SineWave import SineWave

    from torchaudio.transforms import MuLawDecoding
    import torchaudio
    import numpy as np

    arg = {'dilations': [2 ** i for j in range(3) for i in range(10)],
           'kernel_size': 2,
           'block_channels': 32,
           'residual_chanels': 32,
           'skip_channels': 256,
           'end_channels': 256,
           'categories': 256,
           'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}

    net = WaveNet(**arg)

    torch.cuda.empty_cache()

    dataset = PianoDataset('/media/jan//Data/datasets/PianoDataset', batch_size=8, min_length=net.receptive_field, num_targets=8000, num_classes=256)
    #net.load_state_dict(torch.load('wavenet_saved'))

    # dataset = SineWave(net.receptive_field)
    for round in range(5):
        net.train_net(dataset, 5)
        torch.save(net.state_dict(), './model_round_{}'.format(round))

    generator = WaveGenerator(net)
    sound = []
    prgbar = Progbar(8000*3)
    for i in range(8000*3):
        sound.append(next(generator)[0])
        prgbar.update(i)

    from torchaudio.transforms import MuLawDecoding
    decode = MuLawDecoding(256)
    sound = decode(torch.as_tensor(sound))
    torchaudio.save('sine_wave.wav', sound, 8000)
