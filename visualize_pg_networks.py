"""
Visualize P & G network outputs (allocation and payment patterns)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_pg_network_outputs(pg_outputs: List[Dict], n_agents: int, save_path: str = 'graphs/pg_network_outputs.png'):
    """
    Visualize P & G network outputs: who wins, allocation probabilities, and payments
    
    Args:
        pg_outputs: List of dictionaries with P&G outputs from each round
        n_agents: Number of agents
        save_path: Path to save the plot
    """
    if len(pg_outputs) == 0:
        print("No P&G outputs to visualize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    rounds = [out['round'] for out in pg_outputs]
    winners = [out['winner_idx'] for out in pg_outputs]
    payments = [out['payments'] for out in pg_outputs]
    winning_payments = [out['winning_payment'] for out in pg_outputs]
    winning_bids = [out['winning_bid'] for out in pg_outputs]
    payment_ratios = [out['payment_ratio'] for out in pg_outputs]
    revenues = [out['revenue'] for out in pg_outputs]
    
    # 1. Winner distribution (who wins most often)
    ax = axes[0, 0]
    winner_counts = np.bincount(winners, minlength=n_agents)
    ax.bar(range(n_agents), winner_counts, alpha=0.7)
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Number of Wins')
    ax.set_title('Winner Distribution (G Network Output)')
    ax.grid(True, alpha=0.3)
    
    # 2. Payment ratio over time (payment/bid)
    ax = axes[0, 1]
    ax.plot(rounds, payment_ratios, alpha=0.6, linewidth=1)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Payment = Bid (first-price)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Payment / Bid Ratio')
    ax.set_title('Payment Ratio Over Time\n(>1.0 = charging more than bid)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Revenue over time
    ax = axes[0, 2]
    window = min(100, len(revenues) // 10)
    if len(revenues) > window:
        revenue_smooth = np.convolve(revenues, np.ones(window)/window, mode='valid')
        rounds_smooth = rounds[window-1:]
        ax.plot(rounds_smooth, revenue_smooth, linewidth=2)
    else:
        ax.plot(rounds, revenues, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Revenue')
    ax.set_title('Revenue Over Time (P Network Output)')
    ax.grid(True, alpha=0.3)
    
    # 4. Payment vs Bid scatter
    ax = axes[1, 0]
    ax.scatter(winning_bids, winning_payments, alpha=0.5, s=10)
    # Add diagonal line (payment = bid)
    max_val = max(max(winning_bids), max(winning_payments)) if winning_bids and winning_payments else 1.0
    ax.plot([0, max_val], [0, max_val], 'r--', label='Payment = Bid')
    ax.set_xlabel('Winning Bid')
    ax.set_ylabel('Payment Charged')
    ax.set_title('Payment vs Bid (P Network)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Payment distribution by agent (who pays how much)
    ax = axes[1, 1]
    agent_payments = [[] for _ in range(n_agents)]
    for out in pg_outputs:
        for i in range(n_agents):
            if out['payments'][i] > 0:
                agent_payments[i].append(out['payments'][i])
    
    # Box plot of payments by agent
    payment_data = [agent_payments[i] if agent_payments[i] else [0] for i in range(n_agents)]
    ax.boxplot(payment_data, labels=[f'A{i}' for i in range(n_agents)])
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Payment Amount')
    ax.set_title('Payment Distribution by Agent')
    ax.grid(True, alpha=0.3)
    
    # 6. Allocation probability heatmap (if we have allocation data)
    ax = axes[1, 2]
    # Sample allocation probabilities from recent rounds
    sample_size = min(100, len(pg_outputs))
    sample_outputs = pg_outputs[-sample_size:]
    
    # Create heatmap: rounds x agents, showing allocation probabilities
    allocation_matrix = np.array([out['allocation'] for out in sample_outputs])
    
    im = ax.imshow(allocation_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Round (sample)')
    ax.set_ylabel('Agent ID')
    ax.set_title(f'Allocation Probabilities (G Network)\nLast {sample_size} rounds')
    plt.colorbar(im, ax=ax, label='Allocation Probability')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"P&G network visualization saved to {save_path}")
    plt.close()


def print_pg_summary(pg_outputs: List[Dict], n_agents: int):
    """
    Print summary statistics of P & G network outputs
    """
    if len(pg_outputs) == 0:
        print("No P&G outputs to summarize")
        return
    
    print("\n" + "=" * 80)
    print("P & G NETWORK OUTPUT SUMMARY")
    print("=" * 80)
    
    # Winner statistics
    winners = [out['winner_idx'] for out in pg_outputs]
    winner_counts = np.bincount(winners, minlength=n_agents)
    print(f"\nWinner Distribution (G Network):")
    for i in range(n_agents):
        win_rate = winner_counts[i] / len(pg_outputs)
        print(f"  Agent {i}: {winner_counts[i]} wins ({win_rate*100:.1f}%)")
    
    # Payment statistics
    winning_payments = [out['winning_payment'] for out in pg_outputs]
    winning_bids = [out['winning_bid'] for out in pg_outputs]
    payment_ratios = [out['payment_ratio'] for out in pg_outputs]
    
    print(f"\nPayment Statistics (P Network):")
    print(f"  Avg winning bid: {np.mean(winning_bids):.4f}")
    print(f"  Avg payment: {np.mean(winning_payments):.4f}")
    print(f"  Avg payment/bid ratio: {np.mean(payment_ratios):.4f}")
    print(f"  Payment > Bid: {np.sum(np.array(payment_ratios) > 1.0)}/{len(payment_ratios)} rounds ({np.mean(np.array(payment_ratios) > 1.0)*100:.1f}%)")
    print(f"  Max payment: {np.max(winning_payments):.4f}")
    print(f"  Min payment: {np.min(winning_payments):.4f}")
    
    # Revenue statistics
    revenues = [out['revenue'] for out in pg_outputs]
    print(f"\nRevenue Statistics:")
    print(f"  Avg revenue: {np.mean(revenues):.4f}")
    print(f"  Std revenue: {np.std(revenues):.4f}")
    print(f"  Max revenue: {np.max(revenues):.4f}")
    print(f"  Min revenue: {np.min(revenues):.4f}")
    
    print("=" * 80)

