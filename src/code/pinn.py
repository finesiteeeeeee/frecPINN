
import torch
import torch.nn as nn
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # 定义网络
        self.mlp = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Tanh(),
            nn.Linear(layers[1], layers[2]),
            nn.Tanh(),
            nn.Linear(layers[2], layers[3]),
            nn.Tanh(),
            nn.Linear(layers[3], layers[4]),
            nn.Tanh(),
            nn.Linear(layers[4], layers[5]),
        )
        self.initial()

    def initial(self):
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=5 / 3)
                nn.init.constant_(layer.bias, 0.)


    def forward(self, x):
        x = self.mlp(x)
        return x
    