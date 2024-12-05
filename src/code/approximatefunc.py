import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


np.random.seed(1234)
torch.manual_seed(1234)



def func(x):
    f = np.zeros(len(x))
    f = np.reshape(f, (-1, 1))
    for i in range(len(x)):
        if x[i] <= 0:
            f[i] = 0.2 * np.sin(6 * x[i])
        else:
            f[i] = 0.1 * x[i] * np.cos(18 * x[i]) + 1
    return f

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
        self.a = torch.ones(1) / 1000

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




# Define the neural network model
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.L = len(layers) - 1
        self.layers = nn.ModuleList()
        for i in range(self.L - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            # Apply non-linear activation functions after each layer
            self.add_module(f"activation_{i}", nn.Tanh())
        # Output layer
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x, a):
        # Apply the layers with activations
        A = x
        for i in range(self.L - 1):
            A = torch.tanh(10 * a[i] * (self.layers[i](A)))
        Y = self.layers[-1](A)
        return Y



N = 300
x = np.linspace(-3, 3, N + 1)
x = np.reshape(x, (-1, 1))
x = torch.tensor(x, dtype=torch.float32)
y = func(x)

layers = [1] + [50] * 4 + [1]


model = frecPINN(layers)


for name, param in model.named_parameters():
    if 'weight' in name:
        torch.nn.init.normal_(param, mean=0., std=0.1)
    elif 'bias' in name:
        torch.nn.init.constant_(param, 0.1)


optimizer = optim.Adam(model.parameters(), lr=0.001)



def loss_fn(y_pred, y_true, a):
    mse = torch.mean((y_pred - y_true) ** 2)
#     regularizer = 1.0 / (torch.mean(torch.exp(torch.mean(a[0])) + torch.exp(torch.mean(a[1])) +
#                                     torch.exp(torch.mean(a[2])) + torch.exp(torch.mean(a[3]))))
#     return mse + regularizer
    return mse





nmax = 25001
n = 0

MSE_hist = []
Sol = []
a_hist = []
a_initial_value = model.a.item()  # 获取初始值
growth_rate = 0.005
# Start training loop
while n <= nmax:
    n += 1

    optimizer.zero_grad()
    if model.a.data < 10.00000:
      new_a_value = a_initial_value * torch.exp(torch.tensor(growth_rate * n))
      model.a.data = new_a_value

    y_pred = model(x)


    loss = loss_fn(y_pred, torch.tensor(y, dtype=torch.float32), [torch.ones_like(x) for _ in range(len(layers) - 1)])


    loss.backward()
    optimizer.step()


    MSE_hist.append(loss.item())
    if n == 2000 or n == 8000 or n == 15000:
        Sol.append(y_pred.detach().numpy())

    if n % 1000 == 0:
        print(f"Steps: {n}, Loss: {loss.item():.3e}")

# Collect results
Solution = np.concatenate(Sol, axis=1)

# Save the MSE history


# Plot results
fig, ax = plt.subplots(figsize=(6, 6))
# plt.figure()
plt.plot(x.numpy()[0:-1], y[0:-1], 'k-', label='Exact')
plt.plot(x.numpy()[0:-1], Solution[0:-1, -1], 'yx-', label='Predicted at Iter = 15000')
plt.plot(x.numpy()[0:-1], Solution[0:-1, 1], 'b-.', label='Predicted at Iter = 8000')
plt.plot(x.numpy()[0:-1], Solution[0:-1, 0], 'r--', label='Predicted at Iter = 2000')
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.legend(loc='upper left')
plt.show()