import torch
import agent
from agent import *


class Environment(nn.Module):
    def __init_(self):
        super(Environment, self).__init__()
        self.actual_v = torch.rand(1)
        self.agents = [PPOAgent(i, self.actual_v) for i in range(10)]

    def step(self):
        bids = [agent.bid() for agent in self.agents]
        winning_bid = max(bids)
        # change this later
        while bids.count(winning_bid) != 1:
            bids = [agent.bid() for agent in self.agents]
            winning_bid = max(bids)

        return winning_bid
