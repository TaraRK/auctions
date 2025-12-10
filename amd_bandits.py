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
    
    Args:
        max_payment_multiplier: Maximum payment as multiple of max_bid.
                                Default None = no constraint (auctioneer can charge anything).
                                Set to 1.0 for first-price style (payment ≤ bid).
                                Set to 1.5 for 1.5× max_bid constraint.
    """
    def __init__(self, n_agents: int, hidden_dim: int = 64, max_payment_multiplier: float = None):
        super(PaymentNetwork, self).__init__()
        self.n_agents = n_agents
        self.max_payment_multiplier = max_payment_multiplier  # None = no constraint (auctioneer can charge anything)
        
        input_dim = n_agents * 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents)
        )
        
        # Initialize final layer to output payments closer to bids initially
        # This helps the network start in a reasonable range instead of near zero
        with torch.no_grad():
            # Initialize final layer bias to ~0.0 (let network learn from scratch)
            # This prevents anchoring payments at a fixed value
            if hasattr(self.net[-1], 'bias') and self.net[-1].bias is not None:
                self.net[-1].bias.fill_(0.0)  # Start at zero, let learning determine optimal payments
            # Scale down weights so initial outputs are small but learnable
            self.net[-1].weight.data *= 0.01  # Smaller initial weights for more gradual learning
    
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
        payments = torch.relu(payments)  # Ensure non-negative payments (only constraint: payments ≥ 0)
       
        # ============================================================
        # PAYMENT CONSTRAINT (Optional)
        # 
        # When max_payment_multiplier = None (default):
        #   - NO UPPER BOUND (infinite/unbounded)
        #   - This entire if block is SKIPPED
        #   - Payments can be any positive value (only limited by network output)
        #   - No cap based on bids, values, or any other amount
        #
        # When max_payment_multiplier is set (e.g., 1.0, 1.5, etc.):
        #   - Payments are capped at: multiplier × max_bid
        #   - Lines 151-156 below execute to enforce this constraint
        # ============================================================
        if self.max_payment_multiplier is not None:
            # This block ONLY executes when max_payment_multiplier is NOT None
            # When None, this entire block is SKIPPED - no constraint applied
            if bids.dim() == 1:
                max_bid = torch.max(bids).item()
            else:
                max_bid = torch.max(bids, dim=-1)[0].item()
            max_payment = max_bid * self.max_payment_multiplier  # Line 155: Only runs if multiplier is set
            payments = torch.clamp(payments, min=0.0, max=max_payment)  # Line 156: Only runs if multiplier is set
        # When max_payment_multiplier = None: The if block above is SKIPPED
        # Result: payments are UNBOUNDED (infinite) - no max_payment constraint
        
        return payments
    
    def compute_payments(self, bids: np.ndarray, winner_idx: int, allocation: np.ndarray, 
                        values: np.ndarray = None) -> np.ndarray:
        """
        Compute payments for all agents
        
        Args:
            bids: Array of bids from all agents
            winner_idx: Index of winning agent
            allocation: Allocation probabilities or binary allocation
            values: Agent values (optional, for individual rationality constraint)
        
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
            
            # ============================================================
            # NO CONSTRAINTS: Let natural dynamics work
            # - If auctioneer charges too high → agents get negative utility → learn to bid lower/avoid
            # - If agents bid too low → they don't win → learn to bid higher
            # - Natural market forces, no artificial constraints
            # ============================================================
            # No payment constraints - network can charge anything
            # Agents will learn naturally: negative utility → adjust strategy
        
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
        allocation_temperature: float = 1.0,
        max_payment_multiplier: float = None  
    ):
        self.n_agents = n_agents
        self.allocation_temperature = allocation_temperature
        
        self.G_network = AllocationNetwork(n_agents, hidden_dim)
        self.P_network = PaymentNetwork(n_agents, hidden_dim, max_payment_multiplier=max_payment_multiplier)
        
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
        
        # No constraints - let natural dynamics work
        # If payment > value, agent gets negative utility and learns to avoid
        payments = self.P_network.compute_payments(bids, winner_idx, allocation_probs, values=None)
        
        utilities = np.zeros(self.n_agents)
        utilities[winner_idx] = values[winner_idx] - payments[winner_idx]
        
       
        revenue = payments[winner_idx] 
        
        if hasattr(self, 'collect_experience'):
            self.collect_experience(bids, allocation_probs, payments, revenue, values)
        
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
                          payments: np.ndarray, revenue: float, values: np.ndarray = None):
        """
        Collect experience for training (REINFORCE)
        
        Args:
            bids: Bids from agents
            allocation_probs: Allocation probabilities from G network
            payments: Payments from P network
            revenue: Revenue from this auction
            values: Agent values (for individual rationality constraint)
        """
        if not hasattr(self, 'experience_buffer'):
            self.experience_buffer = []
        
        # Track average bid level to see if bids are decreasing over time
        avg_bid = np.mean(bids)
        max_bid = np.max(bids)
        
        self.experience_buffer.append({
            'bids': bids.copy(),
            'allocation_probs': allocation_probs.copy(),
            'payments': payments.copy(),
            'revenue': revenue,
            'avg_bid': avg_bid,
            'max_bid': max_bid,
            'values': values.copy() if values is not None else None
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
            # Reward: raw revenue
            # The feedback loop works naturally:
            # - High payments → agents bid lower (in next phase) → revenue decreases (in next training)
            # - Network sees lower revenue when it trains next → learns to charge less
            # - No explicit penalties needed - the natural feedback loop should work
            advantage = revenue - baseline
            
            # Loss from allocation: weighted by allocation probability
            allocation_loss = -torch.sum(allocation_log_probs * allocation_probs) * advantage
            
            # ============================================================
            # PAYMENT LOSS: Learn from feedback loop
            # The network should learn: high payments → lower bids → lower future revenue
            # We use the actual revenue as reward - if payments are too high, 
            # agents will bid lower in next phase, reducing revenue, and network learns
            # ============================================================
            winner_payment = payments_tensor[0, winner_idx]
            
            # Simple REINFORCE: maximize revenue
            # The feedback loop works because:
            # - High payments → agents bid lower → future revenue decreases
            # - Network sees lower revenue in next training phase → learns to charge less
            # - This creates natural learning without explicit penalties
            payment_loss = -winner_payment * advantage
            
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
    agent_type: str = "ppo",  # "ucb", "epsilon_greedy", "regret_matching", or "ppo"
    theta_options: np.ndarray = None,
    lr_auctioneer: float = 1e-3,
    allocation_temperature: float = 1.0,
    alternate_training: bool = True,
    training_interval: int = 50
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
        # Use learning_rate=0.1 and regret_decay=0.999 to slow down convergence
        agents = [RegretMatchingAgent(i, learning_rate=0.1, regret_decay=0.999) for i in range(n_agents)]
        # Note: RegretMatching requires all_bids, so info_type should be FULL_TRANSPARENCY or FULL_REVELATION
        if info_type not in [InformationType.FULL_TRANSPARENCY, InformationType.FULL_REVELATION]:
            print(f"Warning: RegretMatchingAgent requires all_bids. "
                  f"Current info_type={info_type.value} may not work properly.")
    elif agent_type == "ppo":
        from ppo_agent import PPOAgent
        # PPO agents: no budget constraints for single-shot auctions, but keep for compatibility
        agents = [PPOAgent(i, initial_budget=1000.0, total_auctions=n_rounds) for i in range(n_agents)]
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Use 'ucb', 'epsilon_greedy', 'regret_matching', or 'ppo'")
    
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
    training_phase_hist = []  # Track which phase we're in
    strategy_concentration_hist = []  # Track strategy concentration over time
    strategy_entropy_hist = []  # Track strategy entropy over time
    
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
        
        # For agent learning: 
        # - winning_bid is set to payment (what winner actually pays)
        #   because compute_utility uses winning_bid as price_paid
        # - This is correct for AMD where payment ≠ bid (learned payment mechanism)
        winning_payment = outcome.payments[outcome.winner_idx]
        
        agent_outcome = AuctionOutcome(
            winner_idx=outcome.winner_idx,
            winning_bid=winning_payment,  # Use payment (not bid) for compute_utility
            all_bids=bids.copy(),
            utilities=outcome.utilities,  # These are already correct (value - payment)
            agent_info=outcome.agent_info
        )
        
        if train_agents:
            for i, agent in enumerate(agents):
                # For regret matching in AMD: pass auctioneer to compute counterfactual payments
                if agent_type == "regret_matching":
                    agent.update(values[i], thetas[i], agent_outcome, auctioneer=auctioneer, all_values=values)
                else:
                    # PPO and bandit agents use standard update
                    agent.update(values[i], thetas[i], agent_outcome)
        else:
            pass
        
        if not train_agents:
            should_train = (round_idx % training_interval == 0) or (len(auctioneer.experience_buffer) >= training_interval)
            if should_train and len(auctioneer.experience_buffer) > 0:
                # Debug: Check what network is learning
                if round_idx % 5000 == 0:
                    avg_bid_in_buffer = np.mean([np.mean(exp['bids']) for exp in auctioneer.experience_buffer])
                    max_bid_in_buffer = np.mean([np.max(exp['bids']) for exp in auctioneer.experience_buffer])
                    avg_payment_in_buffer = np.mean([exp['payments'][exp['winner_idx']] for exp in auctioneer.experience_buffer])
                    avg_revenue_in_buffer = np.mean([exp['revenue'] for exp in auctioneer.experience_buffer])
                    payment_ratio = avg_payment_in_buffer / avg_bid_in_buffer if avg_bid_in_buffer > 0 else 0
                    print(f"  [Auctioneer Training @ Round {round_idx}]")
                    print(f"    Avg bid in buffer: {avg_bid_in_buffer:.4f}, Max bid: {max_bid_in_buffer:.4f}")
                    print(f"    Avg payment in buffer: {avg_payment_in_buffer:.4f}")
                    print(f"    Payment/Bid ratio: {payment_ratio:.2f}x")
                    print(f"    Avg revenue in buffer: {avg_revenue_in_buffer:.4f}")
                    print(f"    Buffer size: {len(auctioneer.experience_buffer)}")
                
                loss = auctioneer.train_step()
                auctioneer_loss_hist.append(loss)
        
        training_phase_hist.append(1 if train_agents else 0)
        
        for i in range(n_agents):
            theta_hist[i].append(thetas[i])
        
        # For RegretMatching, compute expected theta from strategy
        if agent_type == "regret_matching":
            expected_thetas = [np.dot(agent.strategy, agent.theta_options) for agent in agents]
            avg_theta_hist.append(np.mean(expected_thetas))
        elif agent_type == "ppo":
            # PPO agents have theta directly
            avg_theta_hist.append(np.mean(thetas))
        else:
            avg_theta_hist.append(np.mean(thetas))
        
        highest_value_idx = np.argmax(values)
        efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)
        
        revenue_hist.append(outcome.revenue)
        allocation_hist.append(outcome.allocation.copy())
        payment_hist.append(outcome.payments.copy())
        
        # Track strategy concentration over time (every 100 rounds for efficiency)
        if round_idx % 100 == 0:
            if agent_type == "regret_matching":
                # Calculate concentration metrics for RegretMatching
                concentrations = []
            elif agent_type == "ppo":
                # PPO: Use variance of theta as concentration metric (lower variance = more concentrated)
                theta_vars = [np.var([t for t in theta_hist[i][-100:]]) if len(theta_hist[i]) >= 100 else 0.0 for i in range(n_agents)]
                concentrations = [1.0 / (1.0 + var) for var in theta_vars]  # Inverse variance as concentration
                entropies = [0.0] * n_agents  # Not applicable for PPO (continuous action space)
                
                strategy_concentration_hist.append(np.mean(concentrations))
                strategy_entropy_hist.append(np.mean(entropies))
            else:
                # For bandits, track Q-value concentration
                # Use variance of best thetas as proxy for concentration
                best_thetas = []
                for agent in agents:
                    # Get best theta for each value bin
                    best_theta_per_bin = []
                    for state in range(agent.n_value_bins):
                        best_arm = np.argmax(agent.Q[state])
                        best_theta = agent.theta_options[best_arm]
                        best_theta_per_bin.append(best_theta)
                    best_thetas.append(np.mean(best_theta_per_bin))
                
                # Lower variance = more concentrated/symmetric
                theta_variance = np.var(best_thetas)
                # Convert to concentration metric (inverse of variance, normalized)
                concentration = 1.0 / (1.0 + theta_variance * 10)  # Scale factor
                strategy_concentration_hist.append(concentration)
                strategy_entropy_hist.append(theta_variance)  # Store variance as "entropy"
        
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
        'strategy_concentration_hist': strategy_concentration_hist,
        'strategy_entropy_hist': strategy_entropy_hist,
        'final_avg_theta': final_avg_theta,
        'final_efficiency': final_efficiency,
        'final_revenue': final_revenue,
        'agent_type': agent_type  # Store agent type for strategy analysis
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
    
    
    # 5. Strategy concentration over time
    ax = axes[2, 0]
    if 'strategy_concentration_hist' in metrics and len(metrics['strategy_concentration_hist']) > 0:
        concentration_hist = metrics['strategy_concentration_hist']
        # Convert to round indices (every 100 rounds)
        round_indices = np.arange(len(concentration_hist)) * 100
        ax.plot(round_indices, concentration_hist, linewidth=2, label='concentration')
        ax.set_xlabel('round')
        ax.set_ylabel('strategy concentration')
        ax.set_title('Strategy Concentration Over Time\n(1=concentrated, 0=uniform)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        # Add horizontal line at uniform (0)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='uniform')
        ax.legend()
    else:
        # Fallback to final theta distribution if no concentration data
        theta_hist = metrics['theta_hist']
        final_thetas = [hist[-1] for hist in theta_hist if len(hist) > 0]
        if len(final_thetas) > 0:
            ax.hist(final_thetas, bins=20, alpha=0.7)
            ax.axvline(x=(n_agents-1)/n_agents, color='r', linestyle='--', label='theory')
            ax.set_xlabel('final theta')
            ax.set_ylabel('count')
            ax.set_title('Final Theta Distribution')
            ax.legend()
    
    # 6. Final theta distribution (or empty)
    ax = axes[2, 1]
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


def extract_agent_strategies(agents, agent_type: str):
    """
    Extract learned strategies from agents
    
    Args:
        agents: List of agents
        agent_type: "ucb", "epsilon_greedy", or "regret_matching"
    
    Returns:
        Dictionary with strategy information
    """
    strategies = {}
    
    if agent_type in ["ucb", "epsilon_greedy"]:
        # Bandit agents: Extract Q-tables
        for i, agent in enumerate(agents):
            strategies[i] = {
                'type': 'bandit',
                'Q_table': agent.Q.copy(),  # [n_value_bins, n_arms]
                'theta_options': agent.theta_options.copy(),
                'n_value_bins': agent.n_value_bins,
                'best_theta_per_state': []
            }
            
            # For each value bin, find best theta
            for state in range(agent.n_value_bins):
                best_arm = np.argmax(agent.Q[state])
                best_theta = agent.theta_options[best_arm]
                strategies[i]['best_theta_per_state'].append(best_theta)
    
    elif agent_type == "regret_matching":
        # RegretMatching: Extract strategy distribution
        for i, agent in enumerate(agents):
            strategies[i] = {
                'type': 'regret_matching',
                'strategy': agent.strategy.copy(),  # Probability distribution
                'theta_options': agent.theta_options.copy(),
                'expected_theta': np.dot(agent.strategy, agent.theta_options),
                'top_thetas': []
            }
            
            # Get top 5 most likely thetas
            top_indices = np.argsort(agent.strategy)[-5:][::-1]
            for idx in top_indices:
                strategies[i]['top_thetas'].append({
                    'theta': agent.theta_options[idx],
                    'probability': agent.strategy[idx]
                })
    elif agent_type == "ppo":
        # PPO: Extract mean theta from recent history
        for i, agent in enumerate(agents):
            if hasattr(agent, 'history') and len(agent.history) > 0:
                recent_thetas = [h[3] for h in agent.history[-100:]]  # Last 100 thetas
                mean_theta = np.mean(recent_thetas) if recent_thetas else agent.theta
            else:
                mean_theta = agent.theta
            strategies[i] = {
                'type': 'ppo',
                'mean_theta': mean_theta,
                'current_theta': agent.theta
            }
    
    return strategies


def plot_agent_strategies(agents, agent_type: str, n_agents: int, 
                          save_path: str = 'graphs/agent_strategies.png'):
    """
    Visualize learned strategies for all agents
    
    Args:
        agents: List of agents
        agent_type: Type of agents
        n_agents: Number of agents
        save_path: Path to save plot
    """
    strategies = extract_agent_strategies(agents, agent_type)
    
    if agent_type in ["ucb", "epsilon_greedy"]:
        # Bandit strategies: Show Q-tables as heatmaps
        fig, axes = plt.subplots(2, (n_agents + 1) // 2, figsize=(15, 10))
        if n_agents == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, agent in enumerate(agents):
            ax = axes[i]
            Q = strategies[i]['Q_table']
            theta_options = strategies[i]['theta_options']
            
            # Show Q-values as heatmap (value_bin vs theta)
            # For clarity, show every 10th theta option
            theta_indices = np.arange(0, len(theta_options), max(1, len(theta_options) // 50))
            Q_subset = Q[:, theta_indices]
            theta_subset = theta_options[theta_indices]
            
            im = ax.imshow(Q_subset.T, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('Value Bin')
            ax.set_ylabel('Theta Index')
            ax.set_title(f'Agent {i} Q-Table\n(Expected Theta: {np.mean(strategies[i]["best_theta_per_state"]):.3f})')
            ax.set_yticks(range(0, len(theta_subset), max(1, len(theta_subset) // 5)))
            ax.set_yticklabels([f'{theta_subset[j]:.2f}' for j in range(0, len(theta_subset), max(1, len(theta_subset) // 5))])
            plt.colorbar(im, ax=ax, label='Q-value')
        
        # Hide unused subplots
        for i in range(n_agents, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Strategy heatmaps saved to {save_path}")
        plt.close()
    
    elif agent_type == "regret_matching":
        # RegretMatching strategies: Show probability distributions
        fig, axes = plt.subplots(2, (n_agents + 1) // 2, figsize=(15, 10))
        if n_agents == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, agent in enumerate(agents):
            ax = axes[i]
            strategy = strategies[i]['strategy']
            theta_options = strategies[i]['theta_options']
            expected_theta = strategies[i]['expected_theta']
            
            # Plot strategy distribution
            ax.plot(theta_options, strategy, linewidth=2)
            ax.axvline(x=expected_theta, color='r', linestyle='--', 
                      label=f'Expected: {expected_theta:.3f}')
            ax.axvline(x=(n_agents-1)/n_agents, color='g', linestyle='--', 
                      label=f'Theory: {(n_agents-1)/n_agents:.3f}')
            ax.set_xlabel('Theta')
            ax.set_ylabel('Probability')
            ax.set_title(f'Agent {i} Strategy Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_agents, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Strategy distributions saved to {save_path}")
        plt.close()


def print_strategy_summary(agents, agent_type: str, n_agents: int):
    """
    Print summary of learned strategies
    
    Args:
        agents: List of agents
        agent_type: Type of agents
        n_agents: Number of agents
    """
    strategies = extract_agent_strategies(agents, agent_type)
    
    print("\n" + "=" * 80)
    print("LEARNED STRATEGY SUMMARY")
    print("=" * 80)
    
    if agent_type in ["ucb", "epsilon_greedy"]:
        print(f"\nBandit Agents ({agent_type}):")
        print(f"{'Agent':<10} {'Avg Theta':<15} {'Theta Range':<20} {'Top Theta':<15}")
        print("-" * 80)
        
        for i in range(n_agents):
            best_thetas = strategies[i]['best_theta_per_state']
            avg_theta = np.mean(best_thetas)
            theta_range = f"[{np.min(best_thetas):.3f}, {np.max(best_thetas):.3f}]"
            top_theta = np.max(best_thetas)
            print(f"{i:<10} {avg_theta:<15.3f} {theta_range:<20} {top_theta:<15.3f}")
        
        # Show value-dependent strategies for all agents
        print("\n" + "-" * 80)
        print("Value-Dependent Strategies (All Agents):")
        for i in range(n_agents):
            print(f"\nAgent {i}:")
            print(f"  Low values (0.0-0.3):  theta ≈ {np.mean(strategies[i]['best_theta_per_state'][:3]):.3f}")
            print(f"  Mid values (0.4-0.6):  theta ≈ {np.mean(strategies[i]['best_theta_per_state'][4:7]):.3f}")
            print(f"  High values (0.7-1.0): theta ≈ {np.mean(strategies[i]['best_theta_per_state'][7:]):.3f}")
    
    elif agent_type == "regret_matching":
        print("\nRegretMatching Agents:")
        print(f"{'Agent':<10} {'Expected Theta':<20} {'Top 3 Thetas':<50}")
        print("-" * 80)
        
        for i in range(n_agents):
            expected = strategies[i]['expected_theta']
            top_thetas = strategies[i]['top_thetas'][:3]
            top_str = ", ".join([f"θ={t['theta']:.3f}({t['probability']:.2f})" 
                                for t in top_thetas])
            print(f"{i:<10} {expected:<20.3f} {top_str:<50}")
        
        # Show strategy concentration for all agents
        print("\n" + "-" * 80)
        print("Strategy Concentration (Top 5 Thetas - All Agents):")
        print("Note: With 500 theta options, uniform distribution = 1/500 = 0.0020 per theta")
        print("Higher probabilities indicate more concentrated strategies.\n")
        
        for i in range(n_agents):
            strategy = strategies[i]['strategy']
            top_thetas = strategies[i]['top_thetas']
            expected = strategies[i]['expected_theta']
            theta_opts = strategies[i]['theta_options']
            
            # Calculate concentration metrics
            top5_prob_sum = sum([t['probability'] for t in top_thetas[:5]])
            entropy = -np.sum(strategy * np.log(strategy + 1e-10))  # Strategy entropy
            max_entropy = np.log(len(strategy))  # Maximum entropy (uniform)
            concentration = 1 - (entropy / max_entropy)  # 0 = uniform, 1 = concentrated
            
            # Show probability mass in different theta ranges
            low_range = np.sum(strategy[(theta_opts >= 0.0) & (theta_opts < 0.4)])
            mid_range = np.sum(strategy[(theta_opts >= 0.4) & (theta_opts < 0.7)])
            high_range = np.sum(strategy[(theta_opts >= 0.7) & (theta_opts <= 1.0)])
            
            print(f"\nAgent {i}:")
            print(f"  Expected theta: {expected:.3f}")
            print(f"  Top 5 thetas total probability: {top5_prob_sum:.4f} ({top5_prob_sum*100:.1f}%)")
            print(f"  Strategy concentration: {concentration:.4f} (0=uniform, 1=concentrated)")
            print(f"  Probability mass: Low[0.0-0.4)={low_range:.3f}, Mid[0.4-0.7)={mid_range:.3f}, High[0.7-1.0]={high_range:.3f}")
            print("  (If uniform: each range ≈ 0.33)")
            
            for j, top in enumerate(top_thetas[:5], 1):
                print(f"  {j}. θ={top['theta']:.3f}: prob={top['probability']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test the AMD implementation
    n_agents = 10
    n_rounds = 30000  # More rounds for convergence
    theta_options = np.linspace(0.0, 1.0, 1000)
    
    # Create graphs directory if it doesn't exist
    os.makedirs('graphs', exist_ok=True)
    
    print("=" * 60)
    print("Running AMD with Learning Agents (Full Algorithm)")
    print("=" * 60)
    print(f"Agents: {n_agents}")
    print(f"Rounds: {n_rounds}")
    print("Alternating training: Every 50 rounds")
    print("="*60)
    
    # Test with RegretMatching agents (uses revealed information)
    agents, auctioneer, metrics = run_amd_simulation(
        n_agents=n_agents,
        n_rounds=n_rounds,
        info_type=InformationType.FULL_REVELATION,  # PPO can use any info type
        agent_type="ppo",  # Use PPO (much faster than regret matching)
        theta_options=None,  # Not needed for RegretMatching
        alternate_training=True,
        training_interval=50
    )
    
    print("\n" + "=" * 60)
    print("Generating Convergence Plots...")
    print("=" * 60)
    
    # Plot convergence
    plot_amd_convergence(metrics, n_agents, save_path='graphs/amd_convergence.png')
    
    # Extract and visualize strategies
    print("\n" + "=" * 60)
    print("Analyzing Learned Strategies...")
    print("=" * 60)
    
    # Get agent type from metrics or use default
    agent_type_used = metrics.get('agent_type', "regret_matching")
    
    # Print strategy summary
    print_strategy_summary(agents, agent_type=agent_type_used, n_agents=n_agents)
    
    # Plot strategy visualizations
    plot_agent_strategies(agents, agent_type=agent_type_used, n_agents=n_agents,
                         save_path='graphs/agent_strategies.png')
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print("Final Results:")
    print(f"  Average Theta: {metrics['final_avg_theta']:.3f}")
    print(f"  Final Efficiency: {metrics['final_efficiency']:.3f}")
    print(f"  Final Revenue: {metrics['final_revenue']:.3f}")
    print("\nPlot saved to: graphs/amd_convergence.png")

