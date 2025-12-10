import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os

from imperfect_info_auctions import (
    BanditAgent,
    EpsilonGreedyBanditAgent,
    InformationType,
    AuctionOutcome,
    AgentInformation
)
from regret_matching import RegretMatchingAgent
from agent import Agent


@dataclass
class AMDOutcome:
    winner_idx: int
    allocation: np.ndarray 
    payments: np.ndarray  
    revenue: float
    utilities: np.ndarray
    agent_info: Dict[int, AgentInformation]  # Info revealed 


class AllocationNetwork(nn.Module):
    """
    G network: Allocation function
    Input: bids from all agents
    Output: Allocation probabilities (softmax) or binary allocation
    """
    def __init__(self, n_agents: int, hidden_dim: int = 64):
        super(AllocationNetwork, self).__init__()
        self.n_agents = n_agents
        
        self.net = nn.Sequential(
            nn.Linear(n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents)
        )
    
    def forward(self, bids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bids: Tensor of shape (batch_size, n_agents) or (n_agents,)
        Returns:
            allocation_logits: Logits for each agent winning
        """
        if bids.dim() == 1:
            bids = bids.unsqueeze(0)  
        
        logits = self.net(bids)
        return logits
    
    def allocate(self, bids: np.ndarray, temperature: float = 1.0) -> Tuple[int, np.ndarray]:
        """
        Allocate winner based on bids
        
        Args:
            bids: Array of bids from all agents
            temperature: Temperature for softmax (lower = more deterministic)
        
        Returns:
            winner_idx: Index of winning agent
            allocation_probs: Probability distribution over agents
        """
        self.eval()  
        with torch.no_grad(): 
            bids_tensor = torch.FloatTensor(bids)
            logits = self.forward(bids_tensor)
            
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_squeezed = probs.squeeze()
        
        if temperature < 0.1:
            winner_idx = int(probs_squeezed.argmax().item())
        else:
            winner_idx = int(torch.multinomial(probs_squeezed, 1).item())
        
        probs_list = probs_squeezed.cpu().detach().tolist()
        probs_np = np.array(probs_list, dtype=np.float64)
        
        return winner_idx, probs_np


class PaymentNetwork(nn.Module):
    """
    P network: Payment function
    Input: bids, winner index, allocation
    Output: Payment amount for each agent
    """
    def __init__(self, n_agents: int, hidden_dim: int = 64):
        super(PaymentNetwork, self).__init__()
        self.n_agents = n_agents
        
        input_dim = n_agents * 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents)
        )
    
    def forward(self, bids: torch.Tensor, winner_idx: int, allocation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bids: Tensor of shape (batch_size, n_agents) or (n_agents,)
            winner_idx: Index of winning agent
            allocation: Allocation probabilities or binary allocation
        Returns:
            payments: Payment for each agent
        """
        if bids.dim() == 1:
            bids = bids.unsqueeze(0)
            allocation = allocation.unsqueeze(0)
        
        batch_size = bids.shape[0]
        
        winner_one_hot = torch.zeros(batch_size, self.n_agents)
        winner_one_hot[:, winner_idx] = 1.0
        
        x = torch.cat([bids, winner_one_hot, allocation], dim=-1)
        
        payments = self.net(x)
        payments = torch.relu(payments)
       
        if bids.dim() == 1:
            max_bid = torch.max(bids).item()
        else:
            max_bid = torch.max(bids, dim=-1)[0].item()
        max_payment = max_bid * 1.5
        payments = torch.clamp(payments, min=0.0, max=max_payment)
        
        return payments
    
    def compute_payments(self, bids: np.ndarray, winner_idx: int, allocation: np.ndarray) -> np.ndarray:
        """
        Compute payments for all agents
        
        Args:
            bids: Array of bids from all agents
            winner_idx: Index of winning agent
            allocation: Allocation probabilities or binary allocation
        
        Returns:
            payments: Payment amount for each agent
        """
        self.eval()  
        with torch.no_grad(): 
            bids_tensor = torch.FloatTensor(bids)
            allocation_tensor = torch.FloatTensor(allocation)
            
            payments_tensor = self.forward(bids_tensor, winner_idx, allocation_tensor)
            payments_list = payments_tensor.squeeze().cpu().detach().tolist()
            payments = np.array(payments_list, dtype=np.float64)
        
        return payments


class LearningAuctioneer:
    """
    Auctioneer that learns allocation (G) and payment (P) networks
    """
    def __init__(
        self,
        n_agents: int,
        lr: float = 1e-3,
        hidden_dim: int = 64,
        allocation_temperature: float = 1.0
    ):
        self.n_agents = n_agents
        self.allocation_temperature = allocation_temperature
        
        self.G_network = AllocationNetwork(n_agents, hidden_dim)
        self.P_network = PaymentNetwork(n_agents, hidden_dim)
        
        self.optimizer = optim.Adam(
            list(self.G_network.parameters()) + list(self.P_network.parameters()),
            lr=lr
        )
        
        self.history = {
            'revenue': [],
            'regret': [],
            'efficiency': []
        }
    
    def run_auction(
        self,
        bids: np.ndarray,
        values: np.ndarray,
        info_type: InformationType = InformationType.FULL_REVELATION
    ) -> AMDOutcome:
        """
        Run auction using learned G and P networks
        
        Args:
            bids: Bids from all agents
            values: Private values of all agents
            info_type: Type of information to reveal
        
        Returns:
            AMDOutcome with allocation, payments, etc.
        """
        winner_idx, allocation_probs = self.G_network.allocate(bids, self.allocation_temperature)
        
        payments = self.P_network.compute_payments(bids, winner_idx, allocation_probs)
        
        utilities = np.zeros(self.n_agents)
        utilities[winner_idx] = values[winner_idx] - payments[winner_idx]
        
       
        revenue = payments[winner_idx] 
        
        if hasattr(self, 'collect_experience'):
            self.collect_experience(bids, allocation_probs, payments, revenue)
        
        agent_info = self._reveal_information(bids, values, winner_idx, payments, info_type)
        
        return AMDOutcome(
            winner_idx=winner_idx,
            allocation=allocation_probs,
            payments=payments,
            revenue=revenue,
            utilities=utilities,
            agent_info=agent_info
        )
    
    def _reveal_information(
        self,
        bids: np.ndarray,
        values: np.ndarray,
        winner_idx: int,
        payments: np.ndarray,
        info_type: InformationType
    ) -> Dict[int, AgentInformation]:
        """Reveal information to agents based on info_type"""
        agent_info = {}
        winning_bid = bids[winner_idx]
        
        for i in range(self.n_agents):
            won = (i == winner_idx)
            own_bid = bids[i]
            own_value = values[i]
            
            if info_type == InformationType.MINIMAL:
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_bid,
                    own_value=own_value
                )
            elif info_type == InformationType.WINNER:
                if won:
                    losing_bids = bids.copy()
                    losing_bids[i] = -1
                    losing_bids = losing_bids[losing_bids >= 0]
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value,
                        winning_bid=winning_bid,
                        losing_bids=losing_bids
                    )
                else:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value
                    )
            elif info_type == InformationType.LOSER:
                if won:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value
                    )
                else:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value,
                        winning_bid=winning_bid
                    )
            elif info_type == InformationType.FULL_TRANSPARENCY:
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_bid,
                    own_value=own_value,
                    winning_bid=winning_bid,
                    all_bids=bids.copy()
                )
            elif info_type == InformationType.FULL_REVELATION:
                
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_bid,
                    own_value=own_value,
                    winning_bid=winning_bid,
                    all_bids=bids.copy()
                )
        
        return agent_info
    
    def collect_experience(self, bids: np.ndarray, allocation_probs: np.ndarray, 
                          payments: np.ndarray, revenue: float):
        """
        Collect experience for training (REINFORCE)
        
        Args:
            bids: Bids from agents
            allocation_probs: Allocation probabilities from G network
            payments: Payments from P network
            revenue: Revenue from this auction
        """
        if not hasattr(self, 'experience_buffer'):
            self.experience_buffer = []
        
        self.experience_buffer.append({
            'bids': bids.copy(),
            'allocation_probs': allocation_probs.copy(),
            'payments': payments.copy(),
            'revenue': revenue
        })
        
        self.history['revenue'].append(revenue)
    
    def train_step(self, baseline: float = None):
        """
        Perform REINFORCE training step to maximize revenue
        
        Args:
            baseline: Baseline revenue for variance reduction (optional)
        """
        if not hasattr(self, 'experience_buffer') or len(self.experience_buffer) == 0:
            return
        
        # Set networks to training mode
        self.G_network.train()
        self.P_network.train()
        
        # Compute baseline if not provided
        if baseline is None:
            baseline = np.mean([exp['revenue'] for exp in self.experience_buffer])
        
        total_loss = 0.0
        
        for exp in self.experience_buffer:
            bids = torch.FloatTensor(exp['bids']).unsqueeze(0)
            allocation_probs = torch.FloatTensor(exp['allocation_probs']).unsqueeze(0)
            revenue = exp['revenue']
            
            # Forward pass through G network
            logits = self.G_network.forward(bids)
            allocation_log_probs = torch.log_softmax(logits / self.allocation_temperature, dim=-1)
            
            # Forward pass through P network (need winner for this)
            winner_idx = int(torch.argmax(allocation_probs, dim=-1).item())
            payments_tensor = self.P_network.forward(bids, winner_idx, allocation_probs)
            
            # REINFORCE loss: -log_prob * (reward - baseline)
            # We want to maximize revenue, so minimize negative revenue
            advantage = revenue - baseline
            
            # Loss from allocation: weighted by allocation probability
            allocation_loss = -torch.sum(allocation_log_probs * allocation_probs) * advantage
            
            # Loss from payment: encourage higher payments (simplified)
            # Payment network should maximize sum of payments
            payment_loss = -torch.sum(payments_tensor) * 0.1  # Scale down payment gradient
            
            loss = allocation_loss + payment_loss
            total_loss += loss
        
        # Average loss
        avg_loss = total_loss / len(self.experience_buffer)
        
        # Backward pass
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.G_network.parameters()) + list(self.P_network.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.G_network.eval()
        self.P_network.eval()
        
        self.experience_buffer = []
        
        return avg_loss.item()


def run_amd_simulation(
    n_agents: int,
    n_rounds: int,
    info_type: InformationType = InformationType.MINIMAL,
    agent_type: str = "regret_matching",  # "ucb", "epsilon_greedy", or "regret_matching"
    theta_options: np.ndarray = None,
    lr_auctioneer: float = 1e-3,
    allocation_temperature: float = 1.0,
    alternate_training: bool = True,
    training_interval: int = 100
):
    """
    Run AMD simulation with learning agents and learning auctioneer
    
    Args:
        n_agents: Number of agents
        n_rounds: Number of auction rounds
        info_type: Type of information revelation
        agent_type: "ucb", "epsilon_greedy", or "regret_matching"
        theta_options: Discretized theta space (only used for bandits)
        lr_auctioneer: Learning rate for auctioneer
        allocation_temperature: Temperature for allocation softmax
    
    Returns:
        agents: List of learning agents
        auctioneer: Learning auctioneer
        metrics: Dictionary of metrics
    """
    # Add compute_utility method to Agent if missing (needed for RegretMatching)
    if not hasattr(Agent, 'compute_utility'):
        def compute_utility(self, value: float, won: bool, price_paid: float) -> float:
            """utility = v - price if won, else 0"""
            return (value - price_paid) if won else 0.0
        Agent.compute_utility = compute_utility
    
    # Create agents based on type
    if agent_type == "ucb":
        if theta_options is None:
            theta_options = np.linspace(0.0, 1.0, 1000)
        agents = [BanditAgent(i, theta_options=theta_options) for i in range(n_agents)]
    elif agent_type == "epsilon_greedy":
        if theta_options is None:
            theta_options = np.linspace(0.0, 1.0, 1000)
        agents = [EpsilonGreedyBanditAgent(i, theta_options=theta_options) for i in range(n_agents)]
    elif agent_type == "regret_matching":
        # RegretMatchingAgent doesn't need theta_options (has its own)
        agents = [RegretMatchingAgent(i) for i in range(n_agents)]
        # Note: RegretMatching requires all_bids, so info_type should be FULL_TRANSPARENCY or FULL_REVELATION
        if info_type not in [InformationType.FULL_TRANSPARENCY, InformationType.FULL_REVELATION]:
            print(f"Warning: RegretMatchingAgent requires all_bids. "
                  f"Current info_type={info_type.value} may not work properly.")
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Use 'ucb', 'epsilon_greedy', or 'regret_matching'")
    
    # Create learning auctioneer
    auctioneer = LearningAuctioneer(n_agents, lr=lr_auctioneer, allocation_temperature=allocation_temperature)
    
    # Tracking
    theta_hist = [[] for _ in range(n_agents)]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    allocation_hist = []
    payment_hist = []
    auctioneer_loss_hist = []
    training_phase_hist = []  
    
    train_agents = True  
    auctioneer.experience_buffer = []
    
    for round_idx in range(n_rounds):
        if alternate_training:
            phase = (round_idx // training_interval) % 2
            train_agents = (phase == 0)
        else:
            train_agents = True 
        
        values = np.array([agent.draw_value() for agent in agents])
        thetas = [agent.choose_theta() for agent in agents]
        bids = np.array([values[i] * thetas[i] for i in range(n_agents)])
        
        outcome = auctioneer.run_auction(bids, values, info_type)
        
        winning_payment = outcome.payments[outcome.winner_idx]
        
        agent_outcome = AuctionOutcome(
            winner_idx=outcome.winner_idx,
            winning_bid=winning_payment,
            all_bids=bids.copy(),
            utilities=outcome.utilities,
            agent_info=outcome.agent_info
        )
        
        if train_agents:
            for i, agent in enumerate(agents):
                agent.update(values[i], thetas[i], agent_outcome)
        else:
            pass
        
        if not train_agents:
            should_train = (round_idx % training_interval == 0) or (len(auctioneer.experience_buffer) >= training_interval)
            if should_train and len(auctioneer.experience_buffer) > 0:
                loss = auctioneer.train_step()
                auctioneer_loss_hist.append(loss)
        
        training_phase_hist.append(1 if train_agents else 0)
        
        for i in range(n_agents):
            theta_hist[i].append(thetas[i])
        
        # For RegretMatching, compute expected theta from strategy
        if agent_type == "regret_matching":
            expected_thetas = [np.dot(agent.strategy, agent.theta_options) for agent in agents]
            avg_theta_hist.append(np.mean(expected_thetas))
        else:
            avg_theta_hist.append(np.mean(thetas))
        
        highest_value_idx = np.argmax(values)
        efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)
        
        revenue_hist.append(outcome.revenue)
        allocation_hist.append(outcome.allocation.copy())
        payment_hist.append(outcome.payments.copy())
        
        if round_idx % 5000 == 0:
            if agent_type == "regret_matching":
                expected_thetas = [np.dot(agent.strategy, agent.theta_options) for agent in agents]
                avg_theta = np.mean(expected_thetas)
            else:
                avg_theta = np.mean(thetas)
            avg_revenue = np.mean(revenue_hist[-1000:]) if len(revenue_hist) > 1000 else np.mean(revenue_hist)
            print(f"Round {round_idx}: avg_theta={avg_theta:.3f}, "
                  f"avg_revenue={avg_revenue:.3f}, "
                  f"efficiency={np.mean(efficiency_hist[-1000:]):.3f}")
    
    # Final results
    print("\n=== AMD Results ===")
    if agent_type == "regret_matching":
        # For RegretMatching, compute expected theta from final strategy
        final_avg_theta = np.mean([np.dot(agent.strategy, agent.theta_options) for agent in agents])
    else:
        final_avg_theta = np.mean([np.mean(hist[-1000:]) for hist in theta_hist if len(hist) > 1000])
    final_efficiency = np.mean(efficiency_hist[-1000:])
    final_revenue = np.mean(revenue_hist[-1000:])
    
    print(f"Final avg theta: {final_avg_theta:.3f}")
    print(f"Final efficiency: {final_efficiency:.3f}")
    print(f"Final revenue: {final_revenue:.3f}")
    
    return agents, auctioneer, {
        'theta_hist': theta_hist,
        'efficiency_hist': efficiency_hist,
        'revenue_hist': revenue_hist,
        'allocation_hist': allocation_hist,
        'payment_hist': payment_hist,
        'auctioneer_loss_hist': auctioneer_loss_hist,
        'training_phase_hist': training_phase_hist,
        'avg_theta_hist': avg_theta_hist,
        'final_avg_theta': final_avg_theta,
        'final_efficiency': final_efficiency,
        'final_revenue': final_revenue
    }


def plot_amd_convergence(metrics: Dict, n_agents: int, save_path: str = 'graphs/amd_convergence.png'):
    """
    Plot convergence graphs for AMD simulation
    
    Args:
        metrics: Dictionary of metrics from run_amd_simulation
        n_agents: Number of agents
        save_path: Path to save the plot
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Theta convergence
    ax = axes[0, 0]
    theta_hist = metrics['theta_hist']
    for i, hist in enumerate(theta_hist):
        if i < 3:  # Only show first 3 for clarity
            ax.plot(hist, alpha=0.3, label=f'agent {i}')
    ax.axhline(y=(n_agents-1)/n_agents, color='r', linestyle='--', 
               label=f'theory (n-1)/n={(n_agents-1)/n_agents:.2f}')
    if 'avg_theta_hist' in metrics:
        ax.plot(metrics['avg_theta_hist'], color='black', linewidth=2, label='avg theta')
    ax.set_xlabel('round')
    ax.set_ylabel('theta (shading factor)')
    ax.set_title('Agent Theta Convergence')
    ax.legend()
    
    # 2. Revenue over time
    ax = axes[0, 1]
    revenue_hist = metrics['revenue_hist']
    window = 100
    if len(revenue_hist) > window:
        revenue_smooth = np.convolve(revenue_hist, np.ones(window)/window, mode='valid')
        ax.plot(revenue_smooth, label='revenue (rolling avg)')
    else:
        ax.plot(revenue_hist, label='revenue')
    ax.set_xlabel('round')
    ax.set_ylabel('revenue')
    ax.set_title('Auctioneer Revenue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Efficiency over time
    ax = axes[1, 0]
    efficiency_hist = metrics['efficiency_hist']
    if len(efficiency_hist) > window:
        efficiency_smooth = np.convolve(efficiency_hist, np.ones(window)/window, mode='valid')
        ax.plot(efficiency_smooth)
    else:
        ax.plot(efficiency_hist)
    ax.axhline(y=1.0, color='r', linestyle='--', label='perfect efficiency')
    ax.set_xlabel('round')
    ax.set_ylabel('fraction efficient (rolling avg)')
    ax.set_title('Auction Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Auctioneer loss (if available)
    ax = axes[1, 1]
    if 'auctioneer_loss_hist' in metrics and len(metrics['auctioneer_loss_hist']) > 0:
        ax.plot(metrics['auctioneer_loss_hist'])
        ax.set_xlabel('training step')
        ax.set_ylabel('loss')
        ax.set_title('Auctioneer Training Loss')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No auctioneer loss data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Auctioneer Training Loss')
    
    
    # 5. Final theta distribution
    ax = axes[2, 0]
    theta_hist = metrics['theta_hist']
    final_thetas = [hist[-1] for hist in theta_hist if len(hist) > 0]
    if len(final_thetas) > 0:
        ax.hist(final_thetas, bins=20, alpha=0.7)
        ax.axvline(x=(n_agents-1)/n_agents, color='r', linestyle='--', label='theory')
        ax.set_xlabel('final theta')
        ax.set_ylabel('count')
        ax.set_title('Final Theta Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nConvergence plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test the AMD implementation
    n_agents = 10
    n_rounds = 20000  # More rounds for convergence
    theta_options = np.linspace(0.0, 1.0, 1000)
    
    # Create graphs directory if it doesn't exist
    os.makedirs('graphs', exist_ok=True)
    
    print("=" * 60)
    print("Running AMD with Learning Agents (Full Algorithm)")
    print("=" * 60)
    print(f"Agents: {n_agents}")
    print(f"Rounds: {n_rounds}")
    print("Alternating training: Every 100 rounds")
    print("="*60)
    
    # Test with RegretMatching agents (uses revealed information)
    agents, auctioneer, metrics = run_amd_simulation(
        n_agents=n_agents,
        n_rounds=n_rounds,
        info_type=InformationType.FULL_REVELATION,  # RegretMatching needs all_bids
        agent_type="regret_matching",  # Use RegretMatching instead of bandits
        theta_options=None,  # Not needed for RegretMatching
        alternate_training=True,
        training_interval=100
    )
    
    print("\n" + "=" * 60)
    print("Generating Convergence Plots...")
    print("=" * 60)
    
    # Plot convergence
    plot_amd_convergence(metrics, n_agents, save_path='graphs/amd_convergence.png')
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print("Final Results:")
    print(f"  Average Theta: {metrics['final_avg_theta']:.3f}")
    print(f"  Final Efficiency: {metrics['final_efficiency']:.3f}")
    print(f"  Final Revenue: {metrics['final_revenue']:.3f}")
    print("\nPlot saved to: graphs/amd_convergence.png")

