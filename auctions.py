import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from matplotlib import pyplot as plt
from regret_matching import RegretMatchingAgent
from q_learning import QLearningAgent
from ppo_agent import PPOAgent

@dataclass
# class AuctionOutcome:
#     winner_idx: int
#     winning_bid: float
#     all_bids: np.ndarray
#     utilities: np.ndarray

@dataclass
class AuctionOutcome:
    winner_idx: int
    winning_bid: float
    all_bids: np.ndarray
    payments: np.ndarray   # what each agent pays (>= 0)
    utilities: np.ndarray  # total utility for each agent for this round

class FirstPriceAuction:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def run_auction(self, values: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
        winner_idx = np.argmax(bids)
        winning_bid = bids[winner_idx]

        payments = np.zeros(self.n_agents)
        utilities = np.zeros(self.n_agents)

        # first-price: winner pays their own bid
        payments[winner_idx] = winning_bid
        utilities[winner_idx] = values[winner_idx] - payments[winner_idx]

        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            payments=payments,
            utilities=utilities,
        )
        
class SecondPriceAuction:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def run_auction(self, values: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
        winner_idx = np.argmax(bids)
        winning_bid = np.sort(bids)[-2] 
        
        payments = np.zeros(self.n_agents)
        utilities = np.zeros(self.n_agents)

        payments[winner_idx] = winning_bid
        utilities[winner_idx] = values[winner_idx] - payments[winner_idx]

        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            payments=payments,
            utilities=utilities,
        )

class AllPayAuction:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def run_auction(self, values: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
        winner_idx = np.argmax(bids)
        winning_bid = bids[winner_idx]

        payments = bids
        # utilities = np.zeros(self.n_agents)

        utilities = values - payments
        # utilities[winner_idx] = values[winner_idx] - payments[winner_idx]

        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            payments=payments,
            utilities=utilities,
        )

class WarOfAttritionAuction:
    """
    War of Attrition Auction (Krishna & Morgan, 1997)

    Mechanism:
    - All agents submit bids (representing "time willing to fight")
    - Highest bidder wins the item
    - Payment rule: Each agent pays for the TIME they stayed in the game
      * If you drop out early (bid < 2nd-highest): pay your own bid
      * If you stay until the end (bid >= 2nd-highest): pay the 2nd-highest bid
      * Equivalent to: pay = min(own_bid, second_highest_bid)

    Payoffs:
    - Winner: value - second_highest_bid (stayed until end)
    - Losers who bid low: -own_bid (dropped out early)
    - Losers who bid high: -second_highest_bid (stayed until game ended)

    Strategic implications:
    - Creates proper incentives: dropping out early costs less
    - Different from all-pay auction (where you always pay your full bid)
    - Equilibrium typically involves mixed strategies

    Reference: Krishna, V., & Morgan, J. (1997). "An Analysis of the War of Attrition
    and the All-Pay Auction." Journal of Economic Theory, 72(2), 343-362.
    """
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def run_auction(self, values: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
        """
        Run a war of attrition auction.

        Args:
            values: Private values of agents
            bids: Bids submitted by agents (representing time/effort willing to expend)

        Returns:
            AuctionOutcome with winner, payments, and utilities
        """
        # Highest bidder wins
        winner_idx = np.argmax(bids)
        winning_bid = bids[winner_idx]

        # Second-highest bid (when game ends - second-to-last player drops out)
        second_highest_bid = np.sort(bids)[-2] if self.n_agents > 1 else bids[winner_idx]

        # Payment rule: Each agent pays for the TIME they stayed in
        # - If you dropped out before game ended (bid < second_highest): pay your own bid
        # - If you stayed until game ended (bid >= second_highest): pay second_highest_bid
        # This is equivalent to: pay min(your_bid, second_highest_bid)
        payments = np.minimum(bids, second_highest_bid)

        # Utilities:
        # - Winner gets value minus payment
        # - All losers pay their payment (negative utility)
        utilities = -payments  # Everyone pays for time they stayed in
        utilities[winner_idx] = values[winner_idx] - payments[winner_idx]  # Winner also gets value

        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            payments=payments,
            utilities=utilities,
        )

# class FirstPriceAuction:
#     def __init__(self, n_agents: int):
#         self.n_agents = n_agents
    
#     def run_auction(self, bids: np.ndarray) -> AuctionOutcome:
#         """run single auction round, return outcome"""
#         winner_idx = np.argmax(bids)
#         winning_bid = bids[winner_idx]
        
#         utilities = np.zeros(self.n_agents)
#         # winner gets value - payment, losers get 0 (already initialized)
        
#         return AuctionOutcome(
#             winner_idx=winner_idx,
#             winning_bid=winning_bid,
#             all_bids=bids.copy(),
#             utilities=utilities  # agents compute their own utility since they know their v
#         )

def plot_results(n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist):
    """plot auction simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ----- theta convergence: mean ± std across agents -----
    ax = axes[0, 0]

    # theta_hist: list of length n_agents, each is [T] over rounds
    theta_arr = np.array(theta_hist)              # shape [n_agents, T]
    T = theta_arr.shape[1]
    rounds = np.arange(T)

    mean_theta = theta_arr.mean(axis=0)           # [T]
    # std_theta = theta_arr.std(axis=0)             # [T]

    # optional: show first few agents as faint lines for intuition
    # for i in range(min(3, n_agents)):
    #     ax.plot(rounds, theta_arr[i], alpha=0.1, linewidth=0.5, label=f'agent {i}' if i == 0 else None)

    # main signal: mean ± std
    ax.plot(rounds, mean_theta, color='black', linewidth=2, label='mean theta')
    # ax.fill_between(rounds, mean_theta - std_theta, mean_theta + std_theta,
    #                 alpha=0.2, label='±1 std')

    theory_theta = (n_agents - 1) / n_agents
    ax.axhline(y=theory_theta, color='r', linestyle='--',
               label=f'theory (n-1)/n={theory_theta:.2f}')

    ax.set_xlabel('round')
    ax.set_ylabel('theta (shading factor)')
    ax.set_title('theta convergence')
    ax.legend()

    # ----- efficiency over time (rolling avg) -----
    ax = axes[0, 1]
    window = 100
    efficiency_smooth = np.convolve(efficiency_hist, np.ones(window) / window, mode='valid')
    ax.plot(efficiency_smooth)
    ax.axhline(y=1.0, color='r', linestyle='--', label='perfect efficiency')
    ax.set_xlabel('round')
    ax.set_ylabel('fraction efficient (rolling avg)')
    ax.set_title('auction efficiency')
    ax.legend()

    # ----- revenue over time (rolling avg) -----
    ax = axes[1, 0]
    revenue_smooth = np.convolve(revenue_hist, np.ones(window) / window, mode='valid')
    ax.plot(revenue_smooth)
    ax.set_xlabel('round')
    ax.set_ylabel('avg revenue')
    ax.set_title('auctioneer revenue')

    # ----- final theta distribution (last 100 rounds per agent) -----
    ax = axes[1, 1]
    tail = 100 if theta_arr.shape[1] >= 100 else theta_arr.shape[1]
    # average theta over last `tail` rounds for each agent
    final_thetas = theta_arr[:, -tail:].mean(axis=1)
    ax.hist(final_thetas, bins=20, alpha=0.7)
    ax.axvline(x=theory_theta, color='r', linestyle='--', label='theory')
    ax.set_xlabel(f'avg theta over last {tail} rounds')
    ax.set_ylabel('count')
    ax.set_title('final theta distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('auction_learning.png', dpi=150)
    plt.show()

    # console summary
    print(f"\nfinal results (n={n_agents}):")
    print(f"theory predicts theta = {theory_theta:.3f}")
    print(f"learned avg theta (last {tail} rounds per agent) = {np.mean(final_thetas):.3f}")
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
        initial_budget=100.0,  # budget for one episode
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
            outcome = auction.run_auction(values, bids)
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
            
            if total_rounds % 100 == 0:
                avg_theta = np.mean([np.mean(agent.history[-100:], axis=0)[3]
                                    for agent in agents if len(agent.history) >= 100])
                theta_variance = np.var([np.mean(agent.history[-100:], axis=0)[3]
                                        for agent in agents if len(agent.history) >= 100])
                # get a random agent's recent bid
                random_agent_idx = np.random.randint(0, n_agents)
                if len(agents[random_agent_idx].history) > 0:
                    recent_bid = agents[random_agent_idx].history[-1][1]  # bid is index 1 in (value, bid, utility, theta)
                    recent_value = agents[random_agent_idx].history[-1][0]
                    print(f"round {total_rounds}: avg_theta={avg_theta:.3f}, var={theta_variance:.4f}, theory={(n_agents - 1)/n_agents:.3f}, agent_{random_agent_idx}_bid={recent_bid:.3f} (v={recent_value:.3f})")
                else:
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
# Commented out to prevent execution on import
# Uncomment if you want to run the simulation directly
# agents = run_simulation(n_agents=10, n_rounds=1000)

