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
    def __init__(self, residual_channels, block_channels, skip_channels, kernel_size, dilation_rate):
        super(WaveBlock, self).__init__()

        self.filter = nn.Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.gate = nn.Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.conv1x1_resid = nn.Conv1d(block_channels, residual_channels, 1)
        self.conv1x1_skip = nn.Conv1d(block_channels, skip_channels, 1)


    def forward(self, residual, pad=True):
        # x has (batch_size, channels, time)
        # ie one sample is a matrix, where each column is next time step and each row is a feature
        # Convolution runs from left to right

        filter = torch.tanh(self.filter(residual))
        gate = torch.sigmoid(self.gate(residual))
        out = filter * gate
        residual = self.conv1x1_resid(out) + residual[..., -out.shape[-1]:]
        skip = self.conv1x1_skip(out)
        return residual, skip


class WaveNet(nn.Module):
    def __init__(self,
                 dilations: list,
                 kernel_size=2,
                 block_channels=32,
                 residual_chanels=32,
                 skip_channels=1024,
                 end_channels=256,
                 categories=256,
                 device='cpu'):
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

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.to(device)
        print('Sending network to ', device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def output_layer(self, x):
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor, probs=False, temperature=1, capture_residuals=False):
        """Forward x through the network

        :param x:
        :param probs:
        :param temperature:
        :param capture_residuals:
        :return:
        """

        if x.shape[-1] < self.receptive_field:
            raise Exception('Expected x.shape[-1]>={} but got x.shape[-1]={}'.format(self.receptive_field, x.shape[-1]))

        x = torch.as_tensor(x)
        x.to(self.device) # send input to device
        resid = self.input_conv(x)
        resids = [resid] if capture_residuals else None

        out = 0
        for block in self.blocks:
            resid, skip = block(resid)
            out = skip + out[:, :, -skip.shape[-1]:] if \
                isinstance(out, torch.Tensor) else skip
            if capture_residuals:
                resids.append(resid)

        out = self.output_layer(out)

        if not self.training and temperature != 1:
            out = out / temperature

        if probs:
            out = self.softmax(out)

        if capture_residuals:
            return out, resids
        else:
            return out

    def train_net(self, dataset, epochs):
        criterion = nn.CrossEntropyLoss()  # only accepts logits!

        for epoch in range(epochs):
            data_loader = iter(DataLoader(dataset, batch_size=8, num_workers=0))
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

        Each item in x should be a list of integers.
        All lists should have the same length.
        :param x: 2D tensor or list of lists
        :return: 3D tensor
        """

        if not isinstance(x, list):
            x = [x]

        if not isinstance(x[0], list):
            x = [x]
        elif isinstance(x[0][0], list):
            raise ValueError('Exceeded max list nesting. x must be a list or a list of lists.')

        return F.one_hot(torch.as_tensor(x), self.categories).type(torch.FloatTensor).permute(0,2,1).to(self.device)

    def time_concat(self, x1, x2):
        return torch.cat((x1, x2), 2)

    def generate(self, timesteps, x: list = None, temperature=1):
        # x a list of integers

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
            yield x.item(), distrib


class WaveGenerator:

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
            out = self.net.output_layer(out)
            distrib = self.net.softmax(out)[0, :, 0]  # flatten

        self.out = self._sample_next(distrib)
        return self.out.item(), distrib


if __name__ == '__main__':
    from data.audio.datasets_interface.PianoDataset import PianoDataset
    from torchaudio.transforms import MuLawDecoding
    import torchaudio

    arg = {'dilations': [2 ** i for j in range(3) for i in range(10)],
           'kernel_size': 2,
           'block_channels': 256,
           'residual_chanels': 256,
           'skip_channels': 256,
           'end_channels': 256,
           'categories': 256,
           'device': 'cuda'}

    net = WaveNet(**arg)

    dataset = PianoDataset('/media/jan//Data/datasets/PianoDataset', batch_size=10, min_length=net.receptive_field, num_targets=32, num_classes=256)
    net.load_state_dict(torch.load('wavenet_saved'))
    for round in range(5):
        net.train_net(dataset, 10)
        torch.save(net.state_dict(), './wavenet_round_{}'.format(round))

    generator = WaveGenerator(net)
    sound = []
    prgbar = Progbar(16000)
    for i in range(16000):
        sound.append(next(generator)[0])
        prgbar.update(i)

    from torchaudio.transforms import MuLawDecoding
    decode = MuLawDecoding(256)
    sound = decode(torch.as_tensor(sound))
    torchaudio.save('amazing_sound.wav', sound, 8000)
