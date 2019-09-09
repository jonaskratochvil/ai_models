import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import torch
from torch.utils.data.dataloader import DataLoader

# FIXME: This sucks. - simply copy the code to my utils..
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.utils import Progbar


# jak jsou organizovane batche v torch?
# N - batch size
# D_in - input dimension
# H - hidden dimension
# D_out - output dimension
# C - number of channels
# L  - length of signal sequence

class WaveBlock(nn.Module):
    def __init__(self, filters,kernel_size, dilation_rate):
        super(WaveBlock, self).__init__()

        # pad left, right, top, bottom
        self.pad = nn.ZeroPad2d((dilation_rate * (kernel_size - 1), 0, 0, 0))
        # how many input features, how many output features
        self.filter = nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate)
        self.gate = nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate)
        self.conv1x1 = nn.Conv1d(filters, filters, 1)

    def forward(self, x):
        # x has (batch_size, channels, time)
        # ie one sample is a matrix, where each column is next time step and each row is a feature
        # Convolution runs from left to right

        x = self.pad(x)
        x = self.filter(x) * self.gate(x)
        x = self.conv1x1(x)
        return x


class WaveNet(nn.Module):
    def __init__(self, filters, kernel_size, dilations, categories, device='cpu'):
        super(WaveNet, self).__init__()

        self.input_conv = nn.Conv1d(categories, filters, kernel_size,1)
        # ModuleList needed - is detected by .cuda() and .to() command
        self.blocks = nn.ModuleList([WaveBlock(filters, kernel_size,d)
                                     for d in dilations])

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(filters, filters, 1)
        self.conv2 = nn.Conv1d(filters, categories, 1)
        # softmax over rows - for each timestep a separate softmax
        # used only optionally in forward
        self.softmax = nn.Softmax(dim=1)

        self.dilations = dilations
        self.filters = filters
        self.categories = categories

        self.device = device
        self.to(device)
        print('Sending network to ', device)

    def forward(self, x, probs=False, temperature=1):

        flow = self.input_conv(x)
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

        if probs:
            out = self.softmax(out)
        return out

    def train_net(self, dataset, epochs):
        criterion = nn.CrossEntropyLoss()  # only accepts logits!
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            data_loader = iter(DataLoader(dataset, batch_size=None, num_workers=1))
            running_loss = 0.0
            prgbar = Progbar(len(data_loader))
            for i, data in enumerate(data_loader):
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                print(inputs.shape, targets.shape)

                optimizer.zero_grad()  # zero the parameter gradients
                outputs = net(inputs, probs=False)  # output logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                prgbar.update(i)

            # FIXME: reportuje porad ten samy loss - co je blbe?
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 330))

    def generate(self, x, timesteps, temperature=1):
        # x is a 2d array (no batch dimension)

        input_len = x.shape[-1]

        result = torch.empty(1, self.categories, timesteps + input_len)
        result[:, :, 0:x.shape[-1]] = x
        result = result.float().to(self.device)

        for i in range(input_len, input_len + timesteps):
            distrib = self.forward(result[:, :, :i], probs=True, temperature=temperature)[0, :, -1]
            result[:, :, i] = F.one_hot(torch.multinomial(distrib, 1),
                                        self.categories)

        # so far returns one-hot encoded outputs
        return result

    def generate_faster(selfx, timesteps, temperature=1):
        # generate by caching already computed activations stored in layerwise queues.
        raise NotImplementedError

    @property
    def receptive_field(self):
        return sum(self.dilations)+1


if __name__ == '__main__':
    from data.audio.datasets_interface.PianoDataset import PianoDataset
    from torchaudio.transforms import MuLawDecoding
    import torchaudio

    dilations = [2**i for j in range(5) for i in range(10)]
    net = WaveNet(filters=15,
                  kernel_size=2,
                  dilations=dilations,
                  categories=256,
                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    print('Net receptive field: {} s'.format(net.receptive_field/24000))

    dataset = PianoDataset('/media/jan//Data/datasets/PianoDataset', batch_size=1, min_audio_length=0.5, max_audio_length=2)
    net.train_net(dataset, 5)
    # TODO: save model regularly and use tensorboard for monitoring
    # print(net.generate(F.one_hot(torch.Tensor([[1,2,3]]).to(torch.int64), num_classes=256).permute(0,2,1), 10))

    """
    sampled = Tensor([torch.multinomial(i, 1) for i in out])
    audio = MuLawDecoding(quantization_channels=256)(sampled)
    torchaudio.save('foo.wav', audio, 44100)
    """

    """
    x = torch.randn(1, 2, 10)
    dilations = [2**i for i in range(4)]
    net = WaveNet(2,2,dilations, 2)
    net.generate(x, 3)
    """



