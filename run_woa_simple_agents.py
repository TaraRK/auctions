"""
Run War of Attrition auction simulation with simple learning agents
"""
import numpy as np
from matplotlib import pyplot as plt
from auctions import WarOfAttritionAuction
from agent import Agent

def run_war_of_attrition_simulation(n_agents: int, n_rounds: int):
    """
    Run War of Attrition auction simulation with simple policy gradient agents.

    Args:
        n_agents: Number of bidding agents
        n_rounds: Number of auction rounds
    """
    auction = WarOfAttritionAuction(n_agents)
    agents = [Agent(i) for i in range(n_agents)]

    # Tracking for plots
    theta_hist = [[] for _ in range(n_agents)]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    winners = [0 for _ in range(n_agents)]

    print(f"Running War of Attrition simulation...")
    print(f"Agents: {n_agents}, Rounds: {n_rounds}")
    print("=" * 60)

    for round_idx in range(n_rounds):
        # Draw values
        values = np.array([agent.draw_value() for agent in agents])
        thetas = [agent.theta for agent in agents]  # Get current theta from each agent
        bids = np.array([agents[i].compute_bid(values[i]) for i in range(n_agents)])

        # Run War of Attrition auction
        outcome = auction.run_auction(values, bids)
        winners[outcome.winner_idx] += 1

        # Agents update
        for i, agent in enumerate(agents):
            utility = outcome.utilities[i]
            agent.update(values[i], bids[i], utility)

        # Tracking
        for i in range(n_agents):
            theta_hist[i].append(agents[i].theta)

        avg_theta_hist.append(np.mean(thetas))

        # Efficiency: did highest-value agent win?
        highest_value_idx = np.argmax(values)
        efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)

        # Revenue: total payment (sum of all payments)
        revenue_hist.append(np.sum(outcome.payments))

        if (round_idx + 1) % 1000 == 0:
            avg_theta = np.mean(thetas)
            theta_variance = np.var(thetas)
            print(f"round {round_idx + 1}: avg_theta={avg_theta:.3f}, var={theta_variance:.4f}, theory={(n_agents - 1)/n_agents:.3f}")

    # Final diagnostics
    print("\n" + "=" * 60)
    print("Final diagnostics:")
    print("=" * 60)
    for i, agent in enumerate(agents):
        if len(agent.history) >= 100:
            avg_utility = np.mean([h[2] for h in agent.history[-100:]])
            win_rate = winners[i]/sum(winners)
            print(f"Agent {i}: avg_utility={avg_utility:.4f}, win_rate={win_rate:.3f}")

    print(f'\nTotal auctions: {sum(winners)}')

    # Plot results
    plot_results_war_of_attrition(n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist)

    return agents


def plot_results_war_of_attrition(n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist):
    """Plot War of Attrition auction simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ----- theta convergence: mean across agents -----
    ax = axes[0, 0]

    theta_arr = np.array(theta_hist)  # shape [n_agents, T]
    T = theta_arr.shape[1]
    rounds = np.arange(T)

    mean_theta = theta_arr.mean(axis=0)  # [T]

    ax.plot(rounds, mean_theta, color='black', linewidth=2, label='mean theta')

    theory_theta = (n_agents - 1) / n_agents
    ax.axhline(y=theory_theta, color='r', linestyle='--',
               label=f'theory (n-1)/n={theory_theta:.2f}')

    ax.set_xlabel('round')
    ax.set_ylabel('theta (shading factor)')
    ax.set_title('War of Attrition - Theta Convergence')
    ax.legend()
    ax.grid(alpha=0.3)

    # ----- efficiency over time (rolling avg) -----
    ax = axes[0, 1]
    window = 100
    efficiency_smooth = np.convolve(efficiency_hist, np.ones(window) / window, mode='valid')
    ax.plot(efficiency_smooth, color='blue')
    ax.axhline(y=1.0, color='r', linestyle='--', label='perfect efficiency')
    ax.set_xlabel('round')
    ax.set_ylabel('fraction efficient (rolling avg)')
    ax.set_title('Auction Efficiency')
    ax.legend()
    ax.grid(alpha=0.3)

    # ----- revenue over time (rolling avg) -----
    ax = axes[1, 0]
    revenue_smooth = np.convolve(revenue_hist, np.ones(window) / window, mode='valid')
    ax.plot(revenue_smooth, color='green')
    ax.set_xlabel('round')
    ax.set_ylabel('total revenue per auction')
    ax.set_title('Auctioneer Revenue (All agents pay 2nd-highest bid)')
    ax.grid(alpha=0.3)

    # ----- final theta distribution (last 100 rounds per agent) -----
    ax = axes[1, 1]
    tail = 100 if theta_arr.shape[1] >= 100 else theta_arr.shape[1]
    final_thetas = theta_arr[:, -tail:].mean(axis=1)
    ax.hist(final_thetas, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=theory_theta, color='r', linestyle='--', label='theory', linewidth=2)
    ax.set_xlabel(f'avg theta over last {tail} rounds')
    ax.set_ylabel('count')
    ax.set_title('Final Theta Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('War of Attrition Auction - Learning Dynamics', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('graphs/auction_war_of_attrition.png', dpi=150)
    print(f"\n✓ Graph saved to: graphs/auction_war_of_attrition.png")

    # Console summary
    print(f"\nFinal results (n={n_agents}):")
    print(f"Theory predicts theta = {theory_theta:.3f}")
    print(f"Learned avg theta (last {tail} rounds per agent) = {np.mean(final_thetas):.3f}")
    print(f"Final efficiency = {np.mean(efficiency_hist[-100:]):.3f}")
    print(f"Avg revenue (last 100) = {np.mean(revenue_hist[-100:]):.3f}")


if __name__ == "__main__":
    # Configuration
    n_agents = 10
    n_rounds = 10000

    print("\n" + "=" * 60)
    print("WAR OF ATTRITION AUCTION SIMULATION")
    print("=" * 60)

    agents = run_war_of_attrition_simulation(n_agents=n_agents, n_rounds=n_rounds)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
