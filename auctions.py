import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from matplotlib import pyplot as plt

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
        bid = value * (1 - chosen_theta)
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

class QLearningAgent(Agent):
    def __init__(
        self,
        agent_id: int,
        n_value_bins: int = 10,
        theta_options: np.ndarray = None,
        alpha: float = 0.1,    
        gamma: float = 0.0,     
        epsilon: float = 0.1    
    ):
        super().__init__(agent_id)
        self.n_value_bins = n_value_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if theta_options is None:
            theta_options = np.linspace(0, 1.0, 500)
        self.theta_options = theta_options

        self.Q = np.zeros((self.n_value_bins, len(self.theta_options)))

        self.current_value = None
        self.current_state_idx = None
        self.current_action_idx = None

        self.history = []  

    def _value_to_state(self, v: float) -> int:
        idx = int(v * self.n_value_bins)
        if idx == self.n_value_bins: 
            idx = self.n_value_bins - 1
        return idx

    def draw_value(self) -> float:
        v = np.random.uniform(0.0, 1.0)
        self.current_value = v
        self.current_state_idx = self._value_to_state(v)
        return v

    def choose_theta(self) -> float:
        assert self.current_state_idx is not None, "call draw_value() before choose_theta()"
        s = self.current_state_idx

        if np.random.rand() < self.epsilon:
            a = np.random.randint(len(self.theta_options))
        else:
            a = np.argmax(self.Q[s])

        self.current_action_idx = a
        theta = self.theta_options[a]
        return theta

    def update(self, value: float, chosen_theta: float, outcome):
        # compute realized utility
        bid = value * (1 - chosen_theta)
        won = (outcome.winner_idx == self.agent_id)
        utility = self.compute_utility(value, won, outcome.winning_bid)

        # log for diagnostics
        self.history.append((value, bid, utility, chosen_theta))

        # Q-learning update
        s = self._value_to_state(value)
        a = self.current_action_idx
        r = utility

        # single-step auction, treat next-state value as irrelevant (gamma=0)
        # target = r + gamma * max_a' Q(s', a')  -> here gamma=0 => target = r
        target = r
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * target
        # if you later want gamma>0, you could sample/approximate s_next here

        # optional: you could adapt epsilon over time if you want annealing
        # self.epsilon = max(self.epsilon * 0.9999, 0.01)

def plot_results(n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist):
    """Plot auction simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # theta convergence
    ax = axes[0, 0]
    for i, hist in enumerate(theta_hist):
        ax.plot(hist, alpha=0.3, label=f'agent {i}' if i < 3 else '')
    ax.axhline(y=(n_agents-1)/n_agents, color='r', linestyle='--', label=f'theory (n-1)/n={(n_agents-1)/n_agents:.2f}')
    ax.plot(avg_theta_hist, color='black', linewidth=2, label='avg theta')
    ax.set_xlabel('round')
    ax.set_ylabel('theta (shading factor)')
    ax.set_title('theta convergence')
    ax.legend()

    # efficiency over time (rolling avg)
    ax = axes[0, 1]
    window = 100
    efficiency_smooth = np.convolve(efficiency_hist, np.ones(window)/window, mode='valid')
    ax.plot(efficiency_smooth)
    ax.axhline(y=1.0, color='r', linestyle='--', label='perfect efficiency')
    ax.set_xlabel('round')
    ax.set_ylabel('fraction efficient (rolling avg)')
    ax.set_title('auction efficiency')
    ax.legend()

    # revenue over time
    ax = axes[1, 0]
    revenue_smooth = np.convolve(revenue_hist, np.ones(window)/window, mode='valid')
    ax.plot(revenue_smooth)
    ax.set_xlabel('round')
    ax.set_ylabel('avg revenue')
    ax.set_title('auctioneer revenue')

    # final theta distribution
    ax = axes[1, 1]
    final_thetas = [hist[-1] for hist in theta_hist]
    ax.hist(final_thetas, bins=20, alpha=0.7)
    ax.axvline(x=(n_agents-1)/n_agents, color='r', linestyle='--', label='theory')
    ax.set_xlabel('final theta')
    ax.set_ylabel('count')
    ax.set_title('final theta distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('auction_learning.png', dpi=150)
    plt.show()

    print(f"\nfinal results (n={n_agents}):")
    print(f"theory predicts theta = {(n_agents-1)/n_agents:.3f}")
    print(f"learned avg theta = {np.mean(final_thetas):.3f}")
    print(f"final efficiency = {np.mean(efficiency_hist[-100:]):.3f}")
    print(f"avg revenue (last 100) = {np.mean(revenue_hist[-100:]):.3f}")

# usage
def run_simulation(n_agents: int, n_rounds: int):
    auction = FirstPriceAuction(n_agents)
    # agents = [Agent(i) for i in range(n_agents)]
    agents = [QLearningAgent(i) for i in range(n_agents)]
    
    # tracking for plots
    theta_hist = [[] for _ in range(n_agents)]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    
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
            if isinstance(agent, RegretMatchingAgent) or isinstance(agent, QLearningAgent):
                agent.update(values[i], thetas[i], outcome)
            else:
                won = (i == outcome.winner_idx)
                price = outcome.winning_bid if won else 0
                utility = agent.compute_utility(values[i], won, price)
                agent.update(values[i], bids[i], utility)
        
        # track metrics
        for i in range(n_agents):
            theta_hist[i].append(thetas[i])
        
        avg_theta_hist.append(np.mean(thetas))
        
        # efficiency: did highest-value agent win?
        highest_value_idx = np.argmax(values)
        efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)
        
        # revenue: winning bid
        revenue_hist.append(outcome.winning_bid)
        
        # log stuff every k rounds
        if round_idx % 5000 == 0:
            if isinstance(agents[0], RegretMatchingAgent):
                # expected theta under current strategy
                avg_thetas = [np.dot(a.strategy, a.theta_options) for a in agents]
                avg_theta = np.mean(avg_thetas)
            else:
                avg_theta = np.mean([a.theta for a in agents])
            print(f"round {round_idx}: avg_theta={avg_theta:.3f}, theory={1/n_agents:.3f}")
            # i think the theory is 1/n, but it could be different! 

    # add after round 49000
    print("\nfinal diagnostics:")
    for i, agent in enumerate(agents):
        avg_utility = np.mean([h[2] for h in agent.history[-1000:]])  # last 1k rounds
        win_rate = np.mean([h[2] > 0 for h in agent.history[-1000:]])
        print(f"agent {i}: avg_utility={avg_utility:.4f}, win_rate={win_rate:.3f}")

    # check strategy concentration
    if isinstance(agent, RegretMatchingAgent):
        print(f"\nagent 0 final strategy (top 5):")
        top_indices = np.argsort(agents[0].strategy)[-5:][::-1]
        for idx in top_indices:
            print(f"  theta={agents[0].theta_options[idx]:.3f}: prob={agents[0].strategy[idx]:.3f}")
        print(f"individual thetas: {[np.dot(a.strategy, a.theta_options) for a in agents]}")
    # else:
    #     print(f"individual thetas: {[np.dot(a.strategy, a.theta_options) for a in agents]}")
    
    # plot results
    plot_results(n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist)
    
    return agents

# test
agents = run_simulation(n_agents=10, n_rounds=50000)

