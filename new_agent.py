import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class Agent(nn.Module):
    def __init__(self, i):
        self.i = i
        self.v_i = torch.rand(1).item()  # randomly initialized
        self.theta_i = torch.rand(1).item()  # randomly initialized
        self.b_i = torch.rand(1).item()  # randomly initialized
        self.lr = 0

    def draw_value(self):
        self.v_i = torch.rand(1).item()
        return self.v_i

    def bid(self):
        self.b_i = self.v_i * (1 - self.theta_i)
        return self.b_i

    def update(self, winning_bid):
        if self.b_i == winning_bid:
            reward = self.v_i - self.b_i
        else:
            reward = 0

        self.theta_i += self.lr * reward

        self.theta_i = float(np.clip(self.theta_i, 0.0, 1.0))

        return reward
