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

from utils.utils import TensorQueue


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

        # pad left, right, top, bottom - False in case we want to do fast generation
        self.pad = nn.ZeroPad2d((dilation_rate * (kernel_size - 1), 0, 0, 0))
        # how many input features, how many output features
        self.filter = nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate)
        self.gate = nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate)
        self.conv1x1 = nn.Conv1d(filters, filters, 1)

    def forward(self, flow, pad=True):
        # x has (batch_size, channels, time)
        # ie one sample is a matrix, where each column is next time step and each row is a feature
        # Convolution runs from left to right

        if pad:
            padded_flow = self.pad(flow)

        out = self.filter(padded_flow) * self.gate(padded_flow)
        out = self.conv1x1(out) # should not need padding
        # return flow output added to flow (residual connection) to form flow to he next block
        # return also out, which goes directly to output layer
        if pad:
            return flow + out, out
        else:
            return flow[-len(out):] + out, out


class WaveNet(nn.Module):
    def __init__(self, filters, kernel_size, dilations, categories, device='cpu'):
        super(WaveNet, self).__init__()

        self.pad = nn.ZeroPad2d((1 * (kernel_size - 1), 0, 0, 0)) # pad before input_conv
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
        self.kernel_size = kernel_size
        self.categories = categories

        self.device = device
        self.to(device)
        print('Sending network to ', device)

    def output_layer(self, x):
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor, probs=False, temperature=1, capture_flows=False):
        """Forward x through the network

        :param x:
        :param probs:
        :param temperature:
        :param capture_flows:
        :return:
        """

        if capture_flows:
            flows=[]

        x = self.pad(x)
        flow = self.input_conv(x)

        if capture_flows:
            flows.append(flow)

        out = 0
        for block in self.blocks:
            flow, block_out = block(flow)
            out += block_out

            if capture_flows:
                flows.append(flow)

        out = self.output_layer(out)

        if not self.training and temperature != 1:
            out = out / temperature

        if probs:
            out = self.softmax(out)

        if capture_flows:
            return out, flows
        else:
            return out

    def forward_flows(self, x, probs=False):
        """Return flow from each layer
        """

        flows = []
        x = self.pad(x)
        flows.append(self.input_conv(x))
        out_layer = 0
        for i, block in enumerate(self.blocks):
            flow, block_out = block(flows[-1])
            out_layer += block_out
            flows.append(flow)

        out_layer = self.activation(out_layer)
        out_layer = self.conv1(out_layer)
        out_layer = self.activation(out_layer)
        out_layer = self.conv2(out_layer)

        if probs:
            out_layer = self.softmax(out_layer)
        return flows, out_layer

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

    def one_hot(self, x: list):
        """

        Each item in x should be a list of integers.
        All lists should have the same length.
        :param x: 2D tensor or list of lists
        :return: 3D tensor
        """
        return F.one_hot(torch.Tensor(x), self.categories)

    def generate(self, x, timesteps, temperature=1):
        # x is a 2d array (no batch dimension) TODO: x should simply be a list of integers..

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

    def generate_faster(self, x,  timesteps, temperature=1):
        """Generate by caching already computed activations stored in layerwise queues.

        X should be a 2D list of integers [[1,2,3,4]] - a list containing one list
        :param x: List of integers
        :param timesteps:
        :param temperature:
        :return:
        """

        inputs = self.one_hot(x)

        # At each `layer` we need to store `layer.dilation` items
        queues = [TensorQueue(max_size) for max_size in self.dilations]
        # for outputs we only need to save the current timestep - 1x1 convolutions
        flows, out = self.forward_flows(inputs, probs=True)

        # push flows to queues
        for q, f in zip(queues, flows):
            q.push(f)

        outputs = []
        for t in range(timesteps):
            output = 0
            new_input = self.one_hot(torch.multinomial(out, 1))
            outputs.append(new_input)
            inputs = torch.cat((inputs, new_input)) # generate next input from out (out is a distribution)
            flow = self.input_conv(inputs[-self.kernel_size:]) # no dilation on input!
            for q, block in zip(queues, self.blocks):
                poped = q.push(flow)
                flow, o = block(torch.cat((poped, flow)))
                output += o # should be output for a single time step
            output = self.output_layer(output)
            out = self.softmax(output)

        return outputs

    @property
    def receptive_field(self):
        return sum(self.dilations)+1

class WaveGenerator:

    def __init__(self, wave_net: WaveNet, x: list):
        self.net = WaveNet
        inputs = self.net.one_hot(x) # tohle je trochu prasarna - poladit

        # At each `layer` we need to store `layer.dilation` items
        self.queues = [TensorQueue(max_size) for max_size in self.net.dilations]
        self.inputs_queue = TensorQueue(1)

        # for outputs we only need to save the current timestep - 1x1 convolutions
        flows, out = self.net.forward_flows(inputs, probs=True)
        out = out[-1] #  FIXME: tady bude problem s dimenzionalitou

        # push flows to queues
        for q, f in zip(self.queues, flows):
            q.push(f)

        self.out = self._sample_next(out)
        self.first_run = True

        # pokracuj zde: asi by bylo dobre udelat typing u cele Wavenety, hlavne na shapy tensoru!
        # roztrhat funkce na mensi kusy a napsat testy na shapes, dtype atd.

    def __iter__(self):
        return self

    def _sample_next(self, distrib) -> int:
        """Sample class from distrib

        :param out:
        :return: sampled class as integer
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
            return self.out

        out = 0
        self.out = self.net.one_hot([self.out])
        # if full, q.push pops necessary items to stay on max cap
        inputs = torch.cat((self.inputs_queue.push(self.out), self.out))
        flow = self.input_conv(inputs)  # no dilation on input!, kernel size 2
        for q, block in zip(self.queues, self.blocks):
            flow, o = block(torch.cat((q.push(flow), flow)))
            out += o # should be output for a single time step
        out = self.output_layer(out)
        out = self.softmax(out)
        self.out = self._sample_next(out)
        return self.out


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



