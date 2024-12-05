import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Gausin activation

class Gausin(nn.Module):
    def __init__(self):
        super(Gausin, self).__init__()
    def forward(self, x):
        return -torch.sin(x) * torch.exp((-x * x) / 2)

class HighfreqSubtractLayer(nn.Module):

    def __init__(self):
        super(HighfreqSubtractLayer, self).__init__()

    def forward(self, x):
        # fourier transform
        x_fourier = torch.fft.fft(x)

        n = x.size(-1)
        freqs = torch.fft.fftfreq(n)

        max_freq = freqs.max().item()

        freq_end = max_freq / 2

        # mask
        mask = (freqs >= freq_end) & (freqs <= max_freq)
        mask = mask.to(x.device)
        x_fourier_masked = x_fourier.masked_fill(mask == False, 0)

        x_real = torch.real(x_fourier_masked)

        return x_real


class LowfreqSubtractLayer(nn.Module):
    def __init__(self):
        super(LowfreqSubtractLayer, self).__init__()

    def forward(self, x):
        x_fourier = torch.fft.fft(x)

        n = x.size(-1)
        freqs = torch.fft.fftfreq(n)

        max_freq = freqs.max().item()

        freq_end = max_freq / 2

        mask = (freqs >= 0) & (freqs <= freq_end)
        mask = mask.to(x.device)
        x_fourier_masked = x_fourier.masked_fill(mask == False, 0)

        x_real = torch.real(x_fourier_masked)

        return x_real


class frecPINN(nn.Module):
    def __init__(self, layers):
        super(frecPINN, self).__init__()
        # initialize a
        self.a = torch.ones(1) / 10000

        self.fr1 = HighfreqSubtractLayer()
        self.fr2 = LowfreqSubtractLayer()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2] * 2, layers[3])
        self.fc4 = nn.Linear(layers[3], layers[4])
        self.fc5 = nn.Linear(layers[4], layers[5])
        self.gaus = Gausin()

    def forward(self, x):
        x = self.gaus(self.fc1(x))

        x_high = self.fr1(x)
        x_low = self.fr2(x)
        x_low = self.gaus(self.fc2(x_low))
        x_high = self.gaus(self.fc2(x_high)) * self.a  # dot a

        # combine low & high
        combined_features = torch.cat((x_low, x_high), dim=1)

        combined_features = self.gaus(self.fc3(combined_features))
        combined_features = self.gaus(self.fc4(combined_features))
        # output = self.fc5(combined_features)
        u = self.fc5(combined_features)

        # u = output[:, 0].unsqueeze(-1)
        # v = output[:, 1].unsqueeze(-1)

        return u
