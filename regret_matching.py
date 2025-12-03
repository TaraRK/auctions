import numpy as np
from agent import Agent

class RegretMatchingAgent(Agent):
    """
    Regret matching guarantees convergence to the set of correlated equilibria, which can be broader than nash.
    """
    def __init__(self, agent_id: int):
        super().__init__(agent_id)
        # discretize theta space
        # self.theta_options = np.linspace(0, 0.9, 20)
        self.theta_options = np.linspace(0, 1.0, 500) # more theta options to choose from
        self.cumulative_regret = np.zeros(len(self.theta_options))
        self.strategy = np.ones(len(self.theta_options)) / len(self.theta_options)
        
    def choose_theta(self) -> float:
        # sample theta according to regret-matched strategy
        # theta is shading factor of their value
        return np.random.choice(self.theta_options, p=self.strategy)

    def update(self, value: float, chosen_theta: float, outcome):
        bid = value * chosen_theta
        won = (outcome.winner_idx == self.agent_id)
        utility = self.compute_utility(value, won, outcome.winning_bid)
        
        # track history for diagnostics
        self.history.append((value, bid, utility, chosen_theta))
        
        # find the highest bid that WASN'T mine
        other_bids = outcome.all_bids.copy()
        other_bids[self.agent_id] = -1  # mask out my bid
        highest_other_bid = np.max(other_bids)
        
        # calculate utility for each alternative theta
        for i, alt_theta in enumerate(self.theta_options):
            alt_bid = value * alt_theta
            
            # would i win against the other bids?
            if alt_bid > highest_other_bid:
                alt_utility = value - alt_bid  # win, pay my bid
            elif alt_bid == highest_other_bid:
                alt_utility = 0.5 * (value - alt_bid)  # tie, 50% chance
            else:
                alt_utility = 0  # lose
            
            regret = alt_utility - utility
            self.cumulative_regret[i] += regret
        
        # update strategy
        positive_regret = np.maximum(self.cumulative_regret, 0)
        if positive_regret.sum() > 0:
            self.strategy = positive_regret / positive_regret.sum()
        else:
            self.strategy = np.ones(len(self.theta_options)) / len(self.theta_options)
