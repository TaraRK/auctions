import numpy as np
from agent import Agent

class RegretMatchingAgent(Agent):
    """
    Regret matching guarantees convergence to the set of correlated equilibria, which can be broader than nash.
    """
    def __init__(self, agent_id: int, learning_rate: float = 1.0, regret_decay: float = 1.0):
        super().__init__(agent_id)
        # discretize theta space
        # self.theta_options = np.linspace(0, 0.9, 20)
        self.theta_options = np.linspace(0, 1.0, 500) # more theta options to choose from
        self.cumulative_regret = np.zeros(len(self.theta_options))
        self.strategy = np.ones(len(self.theta_options)) / len(self.theta_options)
        self.learning_rate = learning_rate  # Learning rate for strategy updates (1.0 = standard regret matching)
        self.regret_decay = regret_decay  # Decay factor for cumulative regret (1.0 = no decay, <1.0 = exponential decay)
        
    def choose_theta(self) -> float:
        # sample theta according to regret-matched strategy
        # theta is shading factor of their value
        return np.random.choice(self.theta_options, p=self.strategy)

    def update(self, value: float, chosen_theta: float, outcome, auctioneer=None, all_values=None):
        """
        Update regret matching strategy
        
        Args:
            value: Agent's private value
            chosen_theta: Theta that was chosen
            outcome: AuctionOutcome with winner, payments, etc.
            auctioneer: Optional auctioneer to compute counterfactual payments (for AMD)
            all_values: Optional array of all agent values (needed for counterfactual payments in AMD)
        """
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
                # In AMD: need to compute what payment auctioneer would charge for this alternative bid
                if auctioneer is not None and all_values is not None:
                    # Create counterfactual bids: replace my bid with alternative bid
                    counterfactual_bids = outcome.all_bids.copy()
                    counterfactual_bids[self.agent_id] = alt_bid
                    
                    # Run counterfactual auction to get payment
                    from imperfect_info_auctions import InformationType
                    counterfactual_outcome = auctioneer.run_auction(
                        counterfactual_bids, all_values, InformationType.MINIMAL
                    )
                    # Only compute payment if I would win
                    if counterfactual_outcome.winner_idx == self.agent_id:
                        alt_payment = counterfactual_outcome.payments[self.agent_id]
                        alt_utility = value - alt_payment
                    else:
                        alt_utility = 0  # Wouldn't win with this bid
                else:
                    # Fallback: assume first-price (payment = bid) if no auctioneer provided
                    alt_utility = value - alt_bid  # win, pay my bid
            elif alt_bid == highest_other_bid:
                # Tie: 50% chance of winning
                if auctioneer is not None and all_values is not None:
                    counterfactual_bids = outcome.all_bids.copy()
                    counterfactual_bids[self.agent_id] = alt_bid
                    counterfactual_outcome = auctioneer.run_auction(
                        counterfactual_bids, all_values, InformationType.MINIMAL
                    )
                    if counterfactual_outcome.winner_idx == self.agent_id:
                        alt_payment = counterfactual_outcome.payments[self.agent_id]
                        alt_utility = 0.5 * (value - alt_payment)
                    else:
                        alt_utility = 0.5 * 0  # 50% chance of losing
                else:
                    alt_utility = 0.5 * (value - alt_bid)  # tie, 50% chance
            else:
                alt_utility = 0  # lose
            
            regret = alt_utility - utility
            self.cumulative_regret[i] += regret
        
        # Apply decay to cumulative regret (helps prevent too-rapid convergence)
        self.cumulative_regret *= self.regret_decay
        
        # update strategy with learning rate
        positive_regret = np.maximum(self.cumulative_regret, 0)
        if positive_regret.sum() > 0:
            new_strategy = positive_regret / positive_regret.sum()
            # Interpolate between old and new strategy using learning rate
            self.strategy = (1 - self.learning_rate) * self.strategy + self.learning_rate * new_strategy
        else:
            # If no positive regret, maintain uniform (but still apply learning rate)
            uniform_strategy = np.ones(len(self.theta_options)) / len(self.theta_options)
            self.strategy = (1 - self.learning_rate) * self.strategy + self.learning_rate * uniform_strategy
