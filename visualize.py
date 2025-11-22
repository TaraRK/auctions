import numpy as np
from auctions import FirstPriceAuction, Agent, RegretMatchingAgent
import matplotlib.pyplot as plt

def run_simulation_with_logging(n_agents: int, n_rounds: int):
    auction = FirstPriceAuction(n_agents)
    agents = [Agent(i, learning_rate=0.01) for i in range(n_agents)]
    
    # tracking
    theta_history = [[] for _ in range(n_agents)]
    avg_theta_history = []
    efficiency_history = []  # did highest-value agent win?
    revenue_history = []
    
    for round_idx in range(n_rounds):
        values = np.array([agent.draw_value() for agent in agents])
        bids = np.array([agents[i].compute_bid(values[i]) for i in range(n_agents)])
        
        outcome = auction.run_auction(bids)
        
        # track metrics
        for i, agent in enumerate(agents):
            won = (i == outcome.winner_idx)
            utility = agent.compute_utility(values[i], won, outcome.winning_bid)
            agent.update(values[i], bids[i], utility)
            theta_history[i].append(agent.theta)
        
        avg_theta_history.append(np.mean([a.theta for a in agents]))
        efficiency_history.append(outcome.winner_idx == np.argmax(values))
        revenue_history.append(outcome.winning_bid)
    
    return agents, theta_history, avg_theta_history, efficiency_history, revenue_history

# run and plot
agents, theta_hist, avg_theta, efficiency, revenue = run_simulation_with_logging(
    n_agents=5, n_rounds=2000
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# theta convergence
ax = axes[0, 0]
for i, hist in enumerate(theta_hist):
    ax.plot(hist, alpha=0.3, label=f'agent {i}')
ax.axhline(y=4/5, color='r', linestyle='--', label='theory (n-1)/n')
ax.plot(avg_theta, color='black', linewidth=2, label='avg theta')
ax.set_xlabel('round')
ax.set_ylabel('theta (shading factor)')
ax.set_title('theta convergence')
ax.legend()

# efficiency over time (rolling avg)
ax = axes[0, 1]
window = 100
efficiency_smooth = np.convolve(efficiency, np.ones(window)/window, mode='valid')
ax.plot(efficiency_smooth)
ax.axhline(y=1.0, color='r', linestyle='--', label='perfect efficiency')
ax.set_xlabel('round')
ax.set_ylabel('fraction efficient (rolling avg)')
ax.set_title('auction efficiency')
ax.legend()

# revenue over time
ax = axes[1, 0]
revenue_smooth = np.convolve(revenue, np.ones(window)/window, mode='valid')
ax.plot(revenue_smooth)
ax.set_xlabel('round')
ax.set_ylabel('avg revenue')
ax.set_title('auctioneer revenue')

# final theta distribution
ax = axes[1, 1]
final_thetas = [hist[-1] for hist in theta_hist]
ax.hist(final_thetas, bins=20, alpha=0.7)
ax.axvline(x=4/5, color='r', linestyle='--', label='theory')
ax.set_xlabel('final theta')
ax.set_ylabel('count')
ax.set_title('final theta distribution')
ax.legend()

plt.tight_layout()
plt.savefig('auction_learning.png', dpi=150)
plt.show()

print(f"\nfinal results (n={len(agents)}):")
print(f"theory predicts theta = {(len(agents)-1)/len(agents):.3f}")
print(f"learned avg theta = {np.mean(final_thetas):.3f}")
print(f"final efficiency = {np.mean(efficiency[-100:]):.3f}")
print(f"avg revenue (last 100) = {np.mean(revenue[-100:]):.3f}")