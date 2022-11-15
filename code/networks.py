import torch.nn as nn
import random


class DQN(nn.Module):
    def __init__(self, actions: int):
        super(DQN, self).__init__()
        # 4 84 * 84
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU()
        )
        # 16 20 * 20
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        # 32 * 9 * 9 -> 256
        self.fc_1 = nn.Sequential(
            nn.Linear(32*9*9, 256),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(256, actions)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(-1)
        x = self.fc_1(x)
        x = self.out_layer(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()
