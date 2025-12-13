import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from auctions import FirstPriceAuction, AuctionOutcome
from ppo_agent import PPOAgent
from agent import Agent


@dataclass
class MultiItemAuctionOutcome:
    """Outcome of a multi-item auction"""
    winners: np.ndarray
    winning_bids: np.ndarray
    all_bids: np.ndarray
    payments: np.ndarray
    utilities: np.ndarray


class FirstPriceMultiItemAuction:
    """
    Simultaneous sealed-bid first-price multi-item auction.
    Each item is auctioned independently using first-price rules.
    Reuses FirstPriceAuction for each item.
    """
    def __init__(self, n_agents: int, n_items: int):
        self.n_agents = n_agents
        self.n_items = n_items
        # Reuse FirstPriceAuction for each item
        self.single_item_auction = FirstPriceAuction(n_agents)

    def run_auction(self, values: np.ndarray, bids: np.ndarray) -> MultiItemAuctionOutcome:
        """
        Run simultaneous auctions for all items.
        
        Args:
            values: [n_agents, n_items] - private values for each agent-item pair
            bids: [n_agents, n_items] - bids from each agent for each item
            
        Returns:
            MultiItemAuctionOutcome with winners, payments, and utilities
        """
        winners = np.zeros(self.n_items, dtype=int)
        winning_bids = np.zeros(self.n_items)
        payments = np.zeros((self.n_agents, self.n_items))
        utilities = np.zeros(self.n_agents)
        
        # For each item, run a first-price auction independently
        for item_idx in range(self.n_items):
            item_values = values[:, item_idx]
            item_bids = bids[:, item_idx]
            
            # Reuse FirstPriceAuction for this item
            outcome = self.single_item_auction.run_auction(item_values, item_bids)
            
            winners[item_idx] = outcome.winner_idx
            winning_bids[item_idx] = outcome.winning_bid
            payments[:, item_idx] = outcome.payments
            utilities += outcome.utilities
        
        return MultiItemAuctionOutcome(
            winners=winners,
            winning_bids=winning_bids,
            all_bids=bids.copy(),
            payments=payments,
            utilities=utilities,
        )


class FactorizedPPOAgent(Agent):
    """
    Agent using factorized PPO for multi-item auctions.
    Maintains independent PPO agents for each item.
    Uses composition to delegate PPO learning to existing PPOAgent code.
    """
    def __init__(
        self, 
        agent_id: int, 
        n_items: int,
        initial_budget: float = 100.0,
        total_auctions: int = 50,
        **ppo_kwargs
    ):
        super().__init__(agent_id)
        self.n_items = n_items
        
        # Create a PPOAgent for each item to handle learning independently
        # All item agents use the same agent_id as the main agent so winner checks work correctly
        self.item_agents = [
            PPOAgent(
                agent_id=agent_id,
                initial_budget=initial_budget,
                total_auctions=total_auctions,
                **ppo_kwargs
            )
            for item_idx in range(n_items)
        ]
    
    def draw_values(self) -> np.ndarray:
        """Draw private values for each item using each item's PPO agent"""
        values = np.zeros(self.n_items)
        for item_idx in range(self.n_items):
            values[item_idx] = self.item_agents[item_idx].draw_value()
        return values
    
    def choose_thetas(self) -> np.ndarray:
        """Sample theta for each item independently using PPOAgent.choose_theta"""
        thetas = np.zeros(self.n_items)
        for item_idx in range(self.n_items):
            # Value already drawn in draw_values, just choose theta
            thetas[item_idx] = self.item_agents[item_idx].choose_theta()
        return thetas
    
    def update(self, values: np.ndarray, chosen_thetas: np.ndarray, 
               outcome: MultiItemAuctionOutcome):
        """
        Update PPO agents for each item independently.
        Reuses PPOAgent.update() by creating single-item outcomes for each item.
        
        Args:
            values: [n_items] - private values for each item
            chosen_thetas: [n_items] - thetas chosen for each item
            outcome: MultiItemAuctionOutcome from the auction
        """
        bids = values * chosen_thetas
        total_utility = 0.0
        
        # Update PPO for each item independently using PPOAgent
        for item_idx in range(self.n_items):
            value = values[item_idx]
            chosen_theta = chosen_thetas[item_idx]
            
            # The PPO agent's current_state should already be set from draw_value()
            item_agent = self.item_agents[item_idx]
            
            # Create a single-item AuctionOutcome for this item to reuse PPOAgent.update()
            item_outcome = AuctionOutcome(
                winner_idx=outcome.winners[item_idx],
                winning_bid=outcome.winning_bids[item_idx],
                all_bids=outcome.all_bids[:, item_idx],
                payments=outcome.payments[:, item_idx],
                utilities=outcome.utilities  # This is total utilities, but item_agent won't use it
            )
            
            # Delegate to PPOAgent.update() - this reuses the existing code directly
            # The PPO agent now has validation built-in to handle NaN/inf values
            item_agent.update(value, chosen_theta, item_outcome)
            
            # Track utility for this item
            won = (outcome.winners[item_idx] == self.agent_id)
            if won:
                total_utility += value - outcome.winning_bids[item_idx]
        
        # Track history (reuse Agent pattern)
        self.history.append((values.copy(), bids.copy(), total_utility, chosen_thetas.copy()))
    
    def reset(self):
        """Reset agent state (useful for episodic simulations)"""
        # Reset all item agents
        for item_agent in self.item_agents:
            item_agent.reset()
        # Clear our own history
        self.history = []


def plot_multi_item_results(n_agents, n_items, theta_hist, efficiency_hist, revenue_hist):
    """Plot multi-item auction simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ----- theta convergence: mean ± std across agents and items -----
    ax = axes[0, 0]

    # theta_hist: list of length n_agents, each is [T, n_items] over rounds
    theta_arr = np.array(theta_hist)  # shape [n_agents, T, n_items]
    T = theta_arr.shape[1]
    rounds = np.arange(T)

    # Average theta across all agents and items
    mean_theta = theta_arr.mean(axis=(0, 2))  # [T] - average across agents and items

    # main signal: mean theta
    ax.plot(rounds, mean_theta, color='black', linewidth=2, label='mean theta')

    theory_theta = (n_agents - 1) / (n_agents)
    ax.axhline(y=theory_theta, color='r', linestyle='--',
               label=f'theory (n-1)/n={theory_theta:.2f}')

    ax.set_xlabel('round')
    ax.set_ylabel('theta (shading factor)')
    ax.set_title('theta convergence (multi-item)')
    ax.legend()
    window = 100

    # ----- revenue over time (rolling avg) -----
    ax = axes[1, 0]
    if len(revenue_hist) >= window:
        revenue_smooth = np.convolve(revenue_hist, np.ones(window) / window, mode='valid')
        ax.plot(revenue_smooth)
    else:
        ax.plot(revenue_hist)
    ax.set_xlabel('round')
    ax.set_ylabel('avg revenue')
    ax.set_title('auctioneer revenue')

    # ----- final theta distribution (last 100 rounds per agent) -----
    ax = axes[1, 1]
    tail = min(100, theta_arr.shape[1])
    # average theta over last `tail` rounds for each agent-item pair
    final_thetas = theta_arr[:, -tail:, :].mean(axis=(1, 2))  # [n_agents]
    ax.hist(final_thetas, bins=20, alpha=0.7)
    ax.axvline(x=theory_theta, color='r', linestyle='--', label='theory')
    ax.set_xlabel(f'avg theta over last {tail} rounds')
    ax.set_ylabel('count')
    ax.set_title('final theta distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('multi_item_auction_learning.png', dpi=150)
    plt.show()

    # console summary
    print(f"\nfinal results (n_agents={n_agents}, n_items={n_items}):")
    print(f"theory predicts theta = {theory_theta:.3f}")
    print(f"learned avg theta (last {tail} rounds per agent) = {np.mean(final_thetas):.3f}")
    print(f"final efficiency = {np.mean(efficiency_hist[-100:]):.3f}")
    print(f"avg revenue (last 100) = {np.mean(revenue_hist[-100:]):.3f}")


def run_multi_item_simulation(n_agents: int, n_items: int, n_rounds: int, auctions_per_episode: int = 50):
    """
    Run simulation of multi-item auctions with factorized PPO agents.
    
    Args:
        n_agents: Number of agents
        n_items: Number of items
        n_rounds: Number of episodes
        auctions_per_episode: Number of auctions per episode (for PPO agents)
    """
    auction = FirstPriceMultiItemAuction(n_agents, n_items)
    agents = [
        FactorizedPPOAgent(
            i, 
            n_items,
            initial_budget=100.0,
            total_auctions=auctions_per_episode
        )
        for i in range(n_agents)
    ]
    
    # Tracking for plots
    total_rounds = 0
    theta_hist = [[] for _ in range(n_agents)]  # Each agent's theta history [T, n_items]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    
    for episode in range(n_rounds):
        # Reset all agents for new episode
        for agent in agents:
            agent.reset()
        
        for round_idx in range(auctions_per_episode):
            # Each agent draws values and chooses thetas for all items
            values = np.array([agent.draw_values() for agent in agents])  # [n_agents, n_items]
            thetas = np.array([agent.choose_thetas() for agent in agents])  # [n_agents, n_items]
            bids = values * thetas  # [n_agents, n_items]
            
            # Run auction
            outcome = auction.run_auction(values, bids)
            
            # Agents update
            for i, agent in enumerate(agents):
                agent.update(values[i], thetas[i], outcome)
            
            # Tracking
            for i in range(n_agents):
                theta_hist[i].append(thetas[i].copy())
            
            avg_theta_hist.append(np.mean(thetas))
            
            # Efficiency: did highest-value agents win each item?
            efficiency = 0.0
            for item_idx in range(n_items):
                highest_value_idx = np.argmax(values[:, item_idx])
                if outcome.winners[item_idx] == highest_value_idx:
                    efficiency += 1.0
            efficiency /= n_items
            efficiency_hist.append(efficiency)
            
            # Revenue: sum of winning bids
            revenue_hist.append(np.sum(outcome.winning_bids))
            
            total_rounds += 1
            
            if total_rounds % 100 == 0:
                avg_theta = np.mean(thetas)
                print(f"round {total_rounds}: efficiency={efficiency:.3f}, "
                      f"revenue={np.sum(outcome.winning_bids):.3f}, "
                      f"avg_theta={avg_theta:.3f}")
    
    # Plot results
    plot_multi_item_results(n_agents, n_items, theta_hist, efficiency_hist, revenue_hist)
    
    return agents, efficiency_hist, revenue_hist, avg_theta_hist


if __name__ == "__main__":
    # Example usage
    agents, efficiency, revenue, theta = run_multi_item_simulation(
        n_agents=10, 
        n_items=5, 
        n_rounds=1000,
        auctions_per_episode=50
    )