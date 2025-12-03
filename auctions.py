import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from matplotlib import pyplot as plt
from regret_matching import RegretMatchingAgent
from q_learning import QLearningAgent
from ppo_agent import PPOAgent

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
    # agents = [QLearningAgent(i) for i in range(n_agents)]
    # agents = [PPOAgent(i, initial_budget=n_rounds * 1000, total_auctions=n_rounds) for i in range(n_agents)]
    auctions_per_episode = 50 # for PPO agents
    agents = [PPOAgent(
        i, 
        initial_budget=50.0,  # budget for one episode
        total_auctions=auctions_per_episode
    ) for i in range(n_agents)]
    
    # tracking for plots
    total_rounds = 0
    theta_hist = [[] for _ in range(n_agents)]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    winners = [0 for _ in range(n_agents)]

        
    for episode in range(n_rounds):
        # reset all agents for new episode
        for agent in agents:
            agent.reset()
        
        for round_idx in range(auctions_per_episode):
            # draw values
            values = np.array([agent.draw_value() for agent in agents])
            thetas = [agent.choose_theta() for agent in agents]
            bids = np.array([values[i] * thetas[i] for i in range(n_agents)])
            
            # run auction
            outcome = auction.run_auction(bids)
            winners[outcome.winner_idx] += 1
            # agents update
            for i, agent in enumerate(agents):
                agent.update(values[i], thetas[i], outcome)
            
            # tracking
            for i in range(n_agents):
                theta_hist[i].append(thetas[i])
            
            avg_theta_hist.append(np.mean(thetas))
            
            # efficiency: did highest-value agent win?
            highest_value_idx = np.argmax(values)
            efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)
            
            # revenue: winning bid
            revenue_hist.append(outcome.winning_bid)
            
            total_rounds += 1
            
            if total_rounds % 1000 == 0:
                avg_theta = np.mean([np.mean(agent.history[-100:], axis=0)[3]
                                    for agent in agents if len(agent.history) >= 100])
                theta_variance = np.var([np.mean(agent.history[-100:], axis=0)[3]
                                        for agent in agents if len(agent.history) >= 100])
                print(f"round {total_rounds}: avg_theta={avg_theta:.3f}, var={theta_variance:.4f}, theory={(n_agents - 1)/n_agents:.3f}")

    # add after round 49000
    print("\nfinal diagnostics:")
    for i, agent in enumerate(agents):
        avg_utility = np.mean([h[2] for h in agent.history[-1000:]])  # last 1k rounds
        win_rate = winners[i]/sum(winners)
        print(f"agent {i}: avg_utility={avg_utility:.4f}, win_rate={win_rate:.3f}")

    print('total sum of winners (sanity check):', sum(winners))
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
agents = run_simulation(n_agents=10, n_rounds=1000)

