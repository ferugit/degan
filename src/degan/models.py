import math

import torch
import torch.nn as nn
import torchaudio


class SerializableModule(nn.Module):

    subclasses = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls.subclasses[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, arc):
        if arc not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arc))

        return cls.subclasses[arc]()

    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.zip')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super().__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)

@SerializableModule.register_model('generator')
class Generator(SerializableModule):

    def __init__(self, model_size=64, post_proc_filt_len=512, verbose=False, upsample=True):
        super().__init__()

        num_channels = 1  # c
        latent_dim = 100

        self.model_size = model_size  # d        
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose
        self.fc1 = nn.Linear(latent_dim, 256 * model_size)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4
        self.deconv_1 = Transpose1dLayer(16 * model_size, 8 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(8 * model_size, 4 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(4 * model_size, 2 * model_size, 25, stride, upsample=upsample)
        self.deconv_4 = Transpose1dLayer(2 * model_size, model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Transpose1dLayer(model_size, num_channels, 25, stride, upsample=upsample)

        if post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        x = torch.nn.functional.relu(x)
        if self.verbose:
            print(x.shape)

        x = torch.nn.functional.relu(self.deconv_1(x))
        if self.verbose:
            print(x.shape)

        x = torch.nn.functional.relu(self.deconv_2(x))
        if self.verbose:
            print(x.shape)

        x = torch.nn.functional.relu(self.deconv_3(x))
        if self.verbose:
            print(x.shape)

        x = torch.nn.functional.relu(self.deconv_4(x))
        if self.verbose:
            print(x.shape)

        output = torch.torch.tanh(self.deconv_5(x))

        output = output.squeeze(1)

        return output


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = torch.nn.functional.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = torch.nn.functional.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass


@SerializableModule.register_model('discriminator')
class Discriminator(SerializableModule):
    def __init__(self, model_size=64, shift_factor=2, alpha=0.2, verbose=False):
        super().__init__()

        num_channels = 1

        self.model_size = model_size  # d
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.fc1 = nn.Linear(256 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):

        # add channel dim
        x = x.unsqueeze(1)

        x = torch.nn.functional.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = torch.nn.functional.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = torch.nn.functional.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = torch.nn.functional.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = torch.nn.functional.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        x = x.view(-1, 256 * self.model_size)
        if self.verbose:
            print(x.shape)

        x = self.fc1(x)

        return x