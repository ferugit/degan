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
    def create(cls, arc, num_classes):
        if arc not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arc))

        return cls.subclasses[arc](num_classes)

    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.zip')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))



@SerializableModule.register_model('lenet')
class LeNet(SerializableModule):

    def __init__(self, num_classes):
        super().__init__()

        # Mel-Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=320,
            hop_length=160,
            n_mels=40
        )

        self.features = nn.Sequential(
            nn.InstanceNorm2d(1),
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        f_bins = 40
        t_bins = 151
        f_r = self.__get_size(f_bins)
        t_r = self.__get_size(t_bins)
    
        self.classifier = nn.Sequential(     
            nn.Linear(32 * f_r * t_r, 100),
            nn.Dropout(0.5),
            nn.ReLU(),  
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # [b, ch, t]
        x = self.mel_spectrogram(x) # [b, ch, f_b, t_b]
        x = self.features(x) 
        x = x.view(x.shape[0], -1) # [b, ch*f_b*t_b]
        x = self.classifier(x) # [b, 10]
        return x

    def __get_size(self, in_dim):
        return int(math.floor((((in_dim-4)/2)-4)/2))


@SerializableModule.register_model('crnn')
class CRNN(SerializableModule):
    """
    Model based on Kiran's Sanjeevan model: https://github.com/ksanjeevan/crnn-audio-classification
    """
    def __init__(self, num_classes):
        super().__init__()

        # Mel-Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=320,
            hop_length=160,
            n_mels=40
        )

        # Feature Extraction: CNN
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(3),
            # Conv 2
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            # Conv 3
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        # RNN
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(640, 100),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes),

        )

    def forward(self, x):
        x = x.unsqueeze(1) # [b, ch, t]
        x = self.mel_spectrogram(x) # [b, ch, f_b, t_b]
        x = self.features(x)
        x = x.permute(0, 3, 1, 2) # [b, t_b, ch, f_b]
        x = x.view(x.shape[0], x.shape[1], -1) # [b, t_b, ch*f_b]
        x, _ = self.lstm(x) # [b, t_b, 64]
        x = x.view(x.shape[0], -1) # [b, t_b*64]
        x = self.classifier(x) # [b, 10]
        return x


class SpeechResnetX(SerializableModule):
    """
    ResNetX architecture for Audio Classification.
    """
    def __init__(self, num_classes, n_layers, n_maps, res_pool=None, dilation=True):
        super().__init__()

        self.n_layers = n_layers

        # Fisrt convolution
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if(res_pool):
            self.pool = nn.AvgPool2d(res_pool)

        # Create convolutional layers
        if(dilation):
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        
        # Add Convolutional layers with BatchNormalization
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        
        # Classifier
        self.fc = nn.Linear(n_maps, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension
        for i in range(self.n_layers + 1):
            y = nn.functional.relu(getattr(self, "conv{}".format(i))(x))
            # If average pool after first layer
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.fc(x)


@SerializableModule.register_model('resnet15')
class Resnet15(SerializableModule):
    """
    ResNet15 architecture for WuW detection.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=320,
            hop_length=160,
            n_mels=40
        )
        self.model = SpeechResnetX(num_classes, n_layers=13, n_maps=45, res_pool=None, dilation=True)

    def forward(self, x):
        x = self.mel_spectrogram(x)
        return self.model(x)