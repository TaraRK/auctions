import torch
import new_agent
from new_agent import *


class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()
        self.actual_v = torch.rand(1)
        self.agents = [Agent(i) for i in range(10)]

    def step(self):
        draw_values = [agent.draw_value() for agent in self.agents]
        bids = [agent.bid() for agent in self.agents]
        # print("bids are", bids)
        winning_bid = max(bids)
        # change this later
        while bids.count(winning_bid) != 1:
            bids = [agent.bid() for agent in self.agents]
            winning_bid = max(bids)

        return winning_bid

    def update(self, winning_bid):
        updates = [agent.update(winning_bid) for agent in self.agents]

    def round(self):
        winning_bid = self.step()
        self.update(winning_bid)


auction_marketplace = Environment()

for i in range(20000):
    auction_marketplace.round()

print("Final values are", [agent.theta_i for agent in auction_marketplace.agents])
