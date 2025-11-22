import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AuctionOutcome:
    winner_idx: int
    winning_bid: float
    all_bids: np.ndarray
    utilities: np.ndarray

class FirstPriceAuction:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
    
    def run_auction(self, bids: np.ndarray) -> AuctionOutcome:
        """run single auction round, return outcome"""
        winner_idx = np.argmax(bids)
        winning_bid = bids[winner_idx]
        
        utilities = np.zeros(self.n_agents)
        # winner gets value - payment, losers get 0 (already initialized)
        
        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            utilities=utilities  # agents compute their own utility since they know their v
        )


class Agent:
    def __init__(self, agent_id: int, learning_rate: float = 0.01):
        self.agent_id = agent_id
        self.theta = np.random.uniform(0, 0.5)  # start with some shading
        self.lr = learning_rate
        self.history = []  # track (v, bid, utility, theta) for analysis
        
    def draw_value(self) -> float:
        """draw private value for this auction"""
        return np.random.uniform(0, 1)
    
    def compute_bid(self, value: float) -> float:
        """bid = v * (1 - theta)"""
        return value * (1 - self.theta)
    
    def compute_utility(self, value: float, won: bool, price_paid: float) -> float:
        """utility = v - price if won, else 0"""
        return (value - price_paid) if won else 0.0
    
    def update(self, value: float, bid: float, utility: float):
        """policy gradient update on theta"""
        # gradient of log(policy) w.r.t theta
        # policy is deterministic bid = v(1-theta), so we use score function
        # simplified: gradient points toward increasing utility
        
        # if utility > 0 (won profitably): could shade more (increase theta)
        # if utility = 0 (lost): should shade less (decrease theta)
        # this is rough heuristic, you'll tune this
        
        if utility > 0:
            # won - consider shading more next time
            gradient = value  # direction to increase theta
            self.theta += self.lr * utility * (gradient / (value + 1e-8))
        else:
            # lost - shade less
            gradient = -value
            self.theta -= self.lr * 0.1  # small nudge toward bidding higher
        
        # keep theta in reasonable bounds
        self.theta = np.clip(self.theta, 0, 0.99)
        
        self.history.append((value, bid, utility, self.theta))

class RegretMatchingAgent(Agent):
    """
    egret matching guarantees convergence to the set of correlated equilibria, which can be broader than nash.
    """
    def __init__(self, agent_id: int):
        super().__init__(agent_id)
        # discretize theta space
        # self.theta_options = np.linspace(0, 0.9, 20)
        self.theta_options = np.linspace(0, 0.9, 100) # more theta options to choose from
        self.cumulative_regret = np.zeros(len(self.theta_options))
        self.strategy = np.ones(len(self.theta_options)) / len(self.theta_options)
        
    def choose_theta(self) -> float:
        # sample theta according to regret-matched strategy
        # theta is shading factor of their value
        return np.random.choice(self.theta_options, p=self.strategy)

    # def draw_value(self) -> float:
    #     """draw private value for this auction"""
    #     return np.random.uniform(0, 1)
    
    def update(self, value: float, chosen_theta: float, outcome):
        bid = value * (1 - chosen_theta)
        won = (outcome.winner_idx == self.agent_id)
        utility = self.compute_utility(value, won, outcome.winning_bid)
        
        # find the highest bid that WASN'T mine
        other_bids = outcome.all_bids.copy()
        other_bids[self.agent_id] = -1  # mask out my bid
        highest_other_bid = np.max(other_bids)
        
        # calculate utility for each alternative theta
        for i, alt_theta in enumerate(self.theta_options):
            alt_bid = value * (1 - alt_theta)
            
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

# usage
def run_simulation(n_agents: int, n_rounds: int):
    auction = FirstPriceAuction(n_agents)
    # agents = [Agent(i) for i in range(n_agents)]
    agents = [RegretMatchingAgent(i) for i in range(n_agents)]
    
    for round_idx in range(n_rounds):
        # each agent draws value and computes bid
        values = np.array([agent.draw_value() for agent in agents])
        thetas = [agent.choose_theta() for agent in agents]  # new: sample theta
        bids = np.array([values[i] * (1 - thetas[i]) for i in range(n_agents)])
        # bids = np.array([agents[i].compute_bid(values[i]) for i in range(n_agents)])
        
        # run auction
        outcome = auction.run_auction(bids)
        
        # agents update based on outcome
        for i, agent in enumerate(agents):
            if isinstance(agent, RegretMatchingAgent):
                agent.update(values[i], thetas[i], outcome)
            else:
                won = (i == outcome.winner_idx)
                price = outcome.winning_bid if won else 0
                utility = agent.compute_utility(values[i], won, price)
                agent.update(values[i], bids[i], utility)
        
        # log stuff every k rounds
        if round_idx % 1000 == 0:
            if isinstance(agents[0], RegretMatchingAgent):
                # expected theta under current strategy
                avg_thetas = [np.dot(a.strategy, a.theta_options) for a in agents]
                avg_theta = np.mean(avg_thetas)
            else:
                avg_theta = np.mean([a.theta for a in agents])
            print(f"round {round_idx}: avg_theta={avg_theta:.3f}, theory={(n_agents-1)/n_agents:.3f}")
        
        # if round_idx % 500 == 0:
        #     print(f"\nagent 0 strategy at round {round_idx}:")
        #     for i, (theta_opt, prob) in enumerate(zip(agents[0].theta_options, agents[0].strategy)):
        #         if prob > 0.01:  # only print significant probabilities
        #             print(f"  theta={theta_opt:.2f}: prob={prob:.3f}")
    # add after round 49000
    print("\nfinal diagnostics:")
    for i, agent in enumerate(agents):
        avg_utility = np.mean([h[2] for h in agent.history[-1000:]])  # last 1k rounds
        win_rate = np.mean([h[2] > 0 for h in agent.history[-1000:]])
        print(f"agent {i}: avg_utility={avg_utility:.4f}, win_rate={win_rate:.3f}")

    # check strategy concentration
    print(f"\nagent 0 final strategy (top 5):")
    top_indices = np.argsort(agents[0].strategy)[-5:][::-1]
    for idx in top_indices:
        print(f"  theta={agents[0].theta_options[idx]:.3f}: prob={agents[0].strategy[idx]:.3f}")
    return agents

# test
agents = run_simulation(n_agents=3, n_rounds=50000)