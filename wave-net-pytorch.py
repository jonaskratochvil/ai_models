import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import torch


# jak jsou organizovane batche v torch?
# N - batch size
# D_in - input dimension
# H - hidden dimension
# D_out - output dimension
# C - number of channels
# L  - length of signal sequence

class WaveBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(WaveBlock, self).__init__()

        # pad left, right, top, bottom
        self.pad = nn.ZeroPad2d((dilation_rate * (kernel_size - 1), 0, 0, 0))
        # how many input features, how many output features
        self.filter = nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate)
        self.gate = nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate)
        self.conv1x1 = nn.Conv1d(2,2,1)

    def forward(self, x):
        # x has (batch_size, channels, time)
        # ie one sample is a matrix, where each column is next time step and each row is a feature
        # Convolution runs from left to right

        x = self.pad(x)
        x = self.filter(x) * self.gate(x)
        x = self.conv1x1(x)

        return x


class WaveNet(nn.Module):
    def __init__(self, filters, kernel_size, dilations, categories):
        super(WaveNet, self).__init__()

        self.blocks = [WaveBlock(filters, kernel_size,d)
                       for d in dilations]

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(filters, filters, 1)
        self.conv2 = nn.Conv1d(filters, categories, 1)
        # softmax over rows - for each timestep a separate softmax
        self.softmax = nn.Softmax(dim=1)

        self.dilations = dilations
        self.filters = filters
        self.categories = categories

    def forward(self, x, temperature=1):

        flow = x
        out = 0
        for block in self.blocks:
            conv = block(flow)
            out += conv
            flow += conv

        out = self.activation(out)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)

        if not self.training:
            out = out / temperature

        out = self.softmax(out)
        return out

    def generate(self, x, timesteps, temperature=1):
        # x is a 2d array (no batch dimension)

        input_len = x.shape[-1]

        result = torch.empty(1, self.filters, timesteps + input_len)
        result[0, :, 0:x.shape[-1]] = x

        for i in range(input_len, input_len + timesteps):
            distrib = self.forward(x, temperature=temperature)[0, :, -1]
            result[0, :, i] = F.one_hot(torch.multinomial(distrib, 1),
                                        self.categories)

        return result

    def generate_faster(selfx, timesteps, temperature=1):
        # generate by caching already computed activations stored in layerwise queues.
        raise NotImplementedError

    @property
    def receptive_field(self):
        return sum(self.dilations)+1


if __name__ == '__main__':
    from audio_dataset import PianoDataset
    from torchaudio.transforms import MuLawEncoding

    dilations = [2**i for j in range(5) for i in range(100)]
    net = WaveNet(filters=256,
                  kernel_size=2,
                  dilations=dilations,
                  categories=256)

    print(net.receptive_field/24000)

    dataset = PianoDataset('/Users/vainerj/PianoDataset', download=False,
                           transforms_on_creation=[MuLawEncoding(quantization_channels=256)])

    audio, _, _ = dataset[1]
    # takes really long time for even 2 minutes of audio
    # -> preprocess during dataset creation? Use sparse tensors?
    oh = F.one_hot(audio[0], num_classes=256)
    print('starting forward...')
    oh = oh.unsqueeze(0).permute(0,2,1) # add batch dimension
    net(oh)

    """
    x = torch.randn(1, 2, 10)
    dilations = [2**i for i in range(4)]
    net = WaveNet(2,2,dilations, 2)
    net.generate(x, 3)
    """



