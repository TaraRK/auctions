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
        
        # Check for NaN/inf in input and replace with zeros
        bids = torch.where(torch.isnan(bids) | torch.isinf(bids), 
                          torch.zeros_like(bids), bids)
        
        logits = self.net(bids)
        
        # Check for NaN/inf in output and replace with zeros
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                torch.zeros_like(logits), logits)
        
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
            
            # Check for NaN/inf in logits and replace with zeros
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                    torch.zeros_like(logits), logits)
            
            # Clamp logits to prevent overflow in softmax
            logits = torch.clamp(logits / temperature, min=-50.0, max=50.0)
            
            probs = torch.softmax(logits, dim=-1)
            
            # Check for NaN/inf in probabilities and replace with uniform
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.shape[-1]  # Uniform distribution
            
            # Prevent allocation collapse: enforce minimum probability floor
            # This ensures all agents have at least some chance of winning
            # With 10 agents, min_prob=0.05 means max prob can be at most 0.55 (1 - 9*0.05)
            # This forces more diversity while still allowing preferences
            min_prob = 0.05  # Each agent gets at least 5% probability (encourages more variance)
            n_agents_prob = probs.shape[-1]
            # Scale original probs to leave room for minimum floor, then add floor
            remaining_prob = 1.0 - min_prob * n_agents_prob
            probs = remaining_prob * probs + min_prob  # Mix: (1 - n*min) * original + min for each
            # Should already sum to 1, but renormalize to be safe
            probs = probs / probs.sum()
            
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
        
        # Initialize final layer to output payments in a reasonable range
        # Starting at 0.5 gives the network room to explore both higher and lower payments
        # This is above typical bids (0.2-0.3) but not unreasonably high
        with torch.no_grad():
            # Initialize final layer bias for softplus output
            # softplus(0) ≈ 0.69, softplus(-1) ≈ 0.31, softplus(-2) ≈ 0.13
            # We want to start around 0.2-0.3, so initialize bias to -1.0
            # softplus(-1) ≈ 0.31, which is a good starting point
            if hasattr(self.net[-1], 'bias') and self.net[-1].bias is not None:
                self.net[-1].bias.fill_(-1.0)  # Start around 0.31 after softplus
            # Use moderate initial weights to allow learning in both directions
            self.net[-1].weight.data *= 0.1  # Moderate initial weights for balanced learning
    
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
        # Use softplus to ensure non-negative payments with smooth gradients
        # No upper bound - payments can be any positive value
        # Training signal will naturally discourage excessive payments
        payments = torch.nn.functional.softplus(payments)  # Unbounded positive outputs
       
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
        max_payment_multiplier: float = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 2.0,  # Very strong entropy to encourage diverse allocation
        vf_coef: float = 0.5,
        update_epochs: int = 4,
        buffer_size: int = 200  # Match training_interval so updates happen naturally during each phase
    ):
        self.n_agents = n_agents
        self.allocation_temperature = allocation_temperature
        
        self.G_network = AllocationNetwork(n_agents, hidden_dim)
        self.P_network = PaymentNetwork(n_agents, hidden_dim, max_payment_multiplier=max_payment_multiplier)
        
        # Value network (critic) to estimate expected revenue given bids
        self.value_network = nn.Sequential(
            nn.Linear(n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.buffer_size = buffer_size
        
        # Separate optimizers with different learning rates for stability
        # Payment network gets much lower LR to prevent explosion
        self.optimizer_G = optim.Adam(self.G_network.parameters(), lr=lr)
        self.optimizer_P = optim.Adam(self.P_network.parameters(), lr=lr * 0.05)  # 20x smaller LR for payments (allows more adaptation)
        self.optimizer_V = optim.Adam(self.value_network.parameters(), lr=lr)
        
        self.history = {
            'revenue': [],
            'regret': [],
            'efficiency': []
        }
        
        # Experience buffer for PPO
        self.buffer = {
            'bids': [],
            'allocation_probs': [],
            'allocation_log_probs': [],
            'payments': [],
            'revenues': [],
            'values': []  # Value network estimates
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
        
        # Compute value estimate and allocation log probs for PPO
        bids_tensor = torch.FloatTensor(bids).unsqueeze(0)
        with torch.no_grad():
            value_estimate = self.value_network(bids_tensor).item()
            logits = self.G_network.forward(bids_tensor)
            allocation_log_probs_tensor = torch.log_softmax(logits / self.allocation_temperature, dim=-1)
            # Sum over agents to get total log prob (for PPO)
            allocation_log_prob_sum = torch.sum(allocation_log_probs_tensor * torch.FloatTensor(allocation_probs).unsqueeze(0), dim=-1).item()
        
        # Collect experience for PPO
        if hasattr(self, 'collect_experience'):
            self.collect_experience(bids, allocation_probs, allocation_log_prob_sum, payments, revenue, value_estimate)
        
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
                          allocation_log_prob_sum: float, payments: np.ndarray, 
                          revenue: float, value_estimate: float):
        """
        Collect experience for PPO-style training
        
        Args:
            bids: Bids from agents
            allocation_probs: Allocation probabilities from G network
            allocation_log_probs: Log probabilities of allocation (for PPO)
            payments: Payments from P network
            revenue: Revenue from this auction (reward)
            value_estimate: Value network estimate of expected revenue
        """
        self.buffer['bids'].append(bids.copy())
        self.buffer['allocation_probs'].append(allocation_probs.copy())
        self.buffer['allocation_log_probs'].append(allocation_log_prob_sum)
        self.buffer['payments'].append(payments.copy())
        self.buffer['revenues'].append(revenue)
        self.buffer['values'].append(value_estimate)
        
        self.history['revenue'].append(revenue)
    
    def update_step(self, force_update: bool = False):
        """
        Perform PPO-style update step (called every round)
        Updates when buffer is full or when forced (e.g., at end of phase)
        """
        if len(self.buffer['bids']) < self.buffer_size and not force_update:
            return 0.0
        
        # If forcing update with small buffer, need at least 1 experience
        if len(self.buffer['bids']) == 0:
            return 0.0
        
        # Set networks to training mode
        self.G_network.train()
        self.P_network.train()
        self.value_network.train()
        
        # Convert buffer to tensors
        bids_tensor = torch.FloatTensor(np.array(self.buffer['bids']))  # [T, n_agents]
        allocation_probs_tensor = torch.FloatTensor(np.array(self.buffer['allocation_probs']))  # [T, n_agents]
        old_allocation_log_probs = torch.FloatTensor(self.buffer['allocation_log_probs'])  # [T]
        revenues = torch.FloatTensor(self.buffer['revenues'])  # [T]
        
        # Compute advantages using GAE
        with torch.no_grad():
            # Get value estimates for all states
            value_output = self.value_network(bids_tensor)  # [T, 1]
            # Squeeze only the last dimension to keep batch dimension
            current_values = value_output.squeeze(-1)  # [T] - always 1D even if T=1
            
            # Compute advantages using GAE
            advantages = torch.zeros_like(revenues)
            lastgaelam = 0.0
            for t in reversed(range(len(revenues))):
                if t == len(revenues) - 1:
                    next_value = 0.0  # Terminal state
                else:
                    next_value = current_values[t + 1]
                delta = revenues[t] + self.gamma * next_value - current_values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + current_values
        
        # Standardize advantages (handle single sample case)
        # For single sample, don't center - use raw advantage to preserve learning signal
        # For multiple samples, standardize to stabilize learning
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            # Only compute std if we have more than 1 sample (avoids warning)
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                # Zero std: just center around mean
                advantages = advantages - adv_mean
        else:
            # Single sample: use raw advantage (don't center to zero!)
            # Centering would make payment_loss = -payments * 0 = 0, killing learning signal
            # The raw advantage preserves the learning signal for payment network
            pass  # Keep advantages as-is for single sample
        
        # PPO update epochs
        total_loss = 0.0
        for epoch in range(self.update_epochs):
            # Recompute log probs with current policy
            # Check for NaN/inf in bids before forward pass
            bids_tensor = torch.where(torch.isnan(bids_tensor) | torch.isinf(bids_tensor), 
                                     torch.zeros_like(bids_tensor), bids_tensor)
            
            logits = self.G_network.forward(bids_tensor)  # [T, n_agents]
            
            # Clamp logits to prevent overflow
            logits = torch.clamp(logits / self.allocation_temperature, min=-50.0, max=50.0)
            new_allocation_log_probs = torch.log_softmax(logits, dim=-1)
            
            # Check for NaN/inf in log probs
            if torch.isnan(new_allocation_log_probs).any() or torch.isinf(new_allocation_log_probs).any():
                # If log probs are invalid, skip this epoch
                continue
            
            # Get winner indices and payments
            winner_indices = torch.argmax(allocation_probs_tensor, dim=-1)  # [T]
            payments_tensor = []
            for t in range(len(bids_tensor)):
                winner_idx = int(winner_indices[t].item())
                payment = self.P_network.forward(bids_tensor[t:t+1], winner_idx, allocation_probs_tensor[t:t+1])
                payments_tensor.append(payment[0, winner_idx])
            payments_tensor = torch.stack(payments_tensor)  # [T]
            
            # Allocation loss (PPO clipped)
            # Sum over agents to get total allocation log prob
            allocation_log_prob_sum = torch.sum(new_allocation_log_probs * allocation_probs_tensor, dim=-1)  # [T]
            old_allocation_log_prob_sum = old_allocation_log_probs  # [T]
            
            ratio = (allocation_log_prob_sum - old_allocation_log_prob_sum).exp()
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef)
            allocation_loss = torch.max(pg_loss1, pg_loss2).mean()
            # Scale down allocation loss to balance with entropy/diversity terms
            allocation_loss = 0.1 * allocation_loss  # Reduce allocation loss weight
            
            # Payment loss: maximize revenue weighted by advantages
            # IMPORTANT: tanh normalization ONLY affects training gradients, NOT payment values
            # - Payment network outputs are UNBOUNDED (softplus can output any positive value)
            # - tanh is applied ONLY to advantages in the loss function for gradient stability
            # - Large advantages → tanh scales them to [-1, 1] → stable gradients
            # - But the network can still learn to output any payment value
            # This prevents gradient explosion during training without restricting actual payments
            advantages_normalized = torch.tanh(advantages / 10.0)  # Only normalizes gradients, not payment outputs
            
            # Payment loss with normalized advantages
            payment_loss = -torch.mean(payments_tensor * advantages_normalized)
            
            # Add a soft penalty for very high payments (encourages reasonable payments but doesn't hard-bound)
            # Use a softer penalty that grows more gradually to allow adaptation
            # For payment=50: penalty ≈ 0.1 * 49^2 = 240, still discourages explosion but allows variation
            # The penalty only applies above 1.0, so reasonable payments (<1.0) aren't penalized
            high_payment_penalty = 0.1 * torch.mean(torch.relu(payments_tensor - 1.0) ** 2)  # Moderate penalty
            payment_loss = payment_loss + high_payment_penalty
            
            # Value loss
            new_values = self.value_network(bids_tensor).squeeze(-1)  # [T] - squeeze last dim only
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
            
            # Entropy bonus (encourage exploration in allocation)
            entropy = -torch.sum(allocation_probs_tensor * new_allocation_log_probs, dim=-1).mean()
            
            # Diversity penalty: directly penalize allocation collapse
            # Penalize when max allocation probability is too high (network always picks same agent)
            # Make penalty very aggressive - penalize any concentration above uniform
            max_allocation_prob = torch.max(allocation_probs_tensor, dim=-1)[0]  # [T]
            uniform_prob = 1.0 / self.n_agents  # For 10 agents, uniform = 0.1
            
            # Strong penalty for concentration above uniform
            # Also add penalty for hitting the constraint boundaries (encourages using full range)
            concentration_penalty = 20.0 * torch.mean(torch.relu(max_allocation_prob - uniform_prob) ** 2)  # Stronger penalty
            
            # Penalty for being too close to constraint boundaries (55% max, 5% min with 5% floor)
            # This encourages network to use the full available range, not just hit boundaries
            max_constraint = 0.55  # Maximum allowed with 5% floor
            min_constraint = 0.05  # Minimum floor
            # Penalize when max prob is very close to constraint (within 0.05 of boundary)
            boundary_penalty = 5.0 * torch.mean(torch.relu(max_allocation_prob - (max_constraint - 0.05)) ** 2)
            
            diversity_penalty = concentration_penalty + boundary_penalty
            
            loss = allocation_loss + payment_loss + self.vf_coef * value_loss - self.ent_coef * entropy + diversity_penalty
            total_loss += loss.item()
            
            # Backward pass - compute gradients
            self.optimizer_G.zero_grad()
            self.optimizer_P.zero_grad()
            self.optimizer_V.zero_grad()
            
            loss.backward()
            
            # Gradient clipping for stability (separate for each network)
            torch.nn.utils.clip_grad_norm_(self.G_network.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.P_network.parameters(), max_norm=0.5)  # Tighter clipping for P network
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
            
            # Update separately
            self.optimizer_G.step()
            self.optimizer_P.step()
            self.optimizer_V.step()
        
        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []
        
        self.G_network.eval()
        self.P_network.eval()
        self.value_network.eval()
        
        return total_loss / self.update_epochs


def run_amd_simulation(
    n_agents: int,
    n_rounds: int,
    info_type: InformationType = InformationType.MINIMAL,
    agent_type: str = "ppo",  # "ucb", "epsilon_greedy", "regret_matching", or "ppo"
    theta_options: np.ndarray = None,
    lr_auctioneer: float = 2e-4,  # Reduced to stabilize learning and prevent runaway revenue
    allocation_temperature: float = 2.0,  # Higher temperature = more stochastic allocation (prevents collapse)
    alternate_training: bool = True,
    training_interval: int = 200,  # Increased to allow better feedback loop observation
    update_epochs: int = 4,  # PPO update epochs for auctioneer
    gamma: float = 0.99  # Discount factor for auctioneer
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
        # Pass gamma to agents so they also use the same discount factor
        agents = [PPOAgent(i, initial_budget=1000.0, total_auctions=n_rounds, gamma=gamma) for i in range(n_agents)]
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Use 'ucb', 'epsilon_greedy', 'regret_matching', or 'ppo'")
    
    # Create learning auctioneer with specified parameters
    auctioneer = LearningAuctioneer(
        n_agents, 
        lr=lr_auctioneer, 
        allocation_temperature=allocation_temperature,
        update_epochs=update_epochs,
        gamma=gamma
    )
    
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
    
    # Track P & G network outputs for analysis
    pg_outputs = []  # Store allocation and payment details for each round
    
    # Initialize buffer (no need for old experience_buffer)
    
    for round_idx in range(n_rounds):
        # Alternating training: one side learns, other is frozen
        # But both can update every round (not batched) when it's their turn
        if alternate_training:
            phase = (round_idx // training_interval) % 2
            train_agents = (phase == 0)
            
            # Clear auctioneer buffer when switching to auctioneer phase
            # This ensures auctioneer starts fresh for its learning phase
            if round_idx > 0 and round_idx % training_interval == 0 and not train_agents:
                # Switching to auctioneer phase - clear buffer to start fresh
                for k in auctioneer.buffer:
                    auctioneer.buffer[k] = []
        else:
            train_agents = True  # Both learn if not alternating
        
        values = np.array([agent.draw_value() for agent in agents])
        thetas = [agent.choose_theta() for agent in agents]
        bids = np.array([values[i] * thetas[i] for i in range(n_agents)])
        
        outcome = auctioneer.run_auction(bids, values, info_type)
        
        # Log P & G network outputs (who wins, allocation probs, payments)
        pg_output = {
            'round': round_idx,
            'bids': bids.copy(),
            'values': values.copy(),
            'winner_idx': outcome.winner_idx,
            'allocation': outcome.allocation.copy(),
            'payments': outcome.payments.copy(),
            'revenue': outcome.revenue,
            'winning_bid': bids[outcome.winner_idx],
            'winning_payment': outcome.payments[outcome.winner_idx],
            'payment_ratio': outcome.payments[outcome.winner_idx] / bids[outcome.winner_idx] if bids[outcome.winner_idx] > 0 else 0.0
        }
        pg_outputs.append(pg_output)
        
        # Print P & G outputs periodically (every 1000 rounds)
        if round_idx % 1000 == 0 and round_idx > 0:
            print(f"\n[Round {round_idx}] P & G Network Outputs:")
            print(f"  Winner: Agent {outcome.winner_idx} (bid={bids[outcome.winner_idx]:.4f}, value={values[outcome.winner_idx]:.4f})")
            print(f"  Payment: {outcome.payments[outcome.winner_idx]:.4f} (ratio: {pg_output['payment_ratio']:.2f}x bid)")
            print(f"  Revenue: {outcome.revenue:.4f}")
            print(f"  Allocation probs (top 3): {sorted(zip(range(n_agents), outcome.allocation), key=lambda x: x[1], reverse=True)[:3]}")
        
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
        
        # Update agents (if it's their phase)
        if train_agents:
            for i, agent in enumerate(agents):
                # For regret matching in AMD: pass auctioneer to compute counterfactual payments
                if agent_type == "regret_matching":
                    agent.update(values[i], thetas[i], agent_outcome, auctioneer=auctioneer, all_values=values)
                else:
                    # PPO and bandit agents use standard update
                    agent.update(values[i], thetas[i], agent_outcome)
        
        # Update auctioneer (if it's their phase, PPO-style updates every round)
        if not train_agents:
            # Force update every round during auctioneer phase (we update every round now)
            # Buffer accumulates 1 experience per round, so we need to force update
            force_update = len(auctioneer.buffer['bids']) > 0
            loss = auctioneer.update_step(force_update=force_update)
            # Record loss whenever an update actually happened (loss != 0.0 means update occurred)
            if loss != 0.0:
                auctioneer_loss_hist.append(loss)
        
        # Debug output at intervals (for stability monitoring)
        if round_idx % training_interval == 0 and round_idx > 0:
            if len(auctioneer.buffer['revenues']) > 0:
                avg_revenue = np.mean(auctioneer.buffer['revenues'])
                avg_bid = np.mean([np.mean(b) for b in auctioneer.buffer['bids']]) if len(auctioneer.buffer['bids']) > 0 else 0.0
                print(f"  [Round {round_idx}] Avg revenue: {avg_revenue:.4f}, Avg bid: {avg_bid:.4f}, Buffer size: {len(auctioneer.buffer['bids'])}")
        
        # Track phase for visualization
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
    
    # Convergence metrics
    if len(revenue_hist) > 2000:
        # Compare last 1000 vs previous 1000 rounds
        recent_revenue = np.mean(revenue_hist[-1000:])
        prev_revenue = np.mean(revenue_hist[-2000:-1000])
        revenue_change = abs(recent_revenue - prev_revenue) / (prev_revenue + 1e-6)
        
        recent_theta = np.mean([np.mean(hist[-1000:]) for hist in theta_hist if len(hist) > 1000])
        prev_theta = np.mean([np.mean(hist[-2000:-1000]) for hist in theta_hist if len(hist) > 2000])
        theta_change = abs(recent_theta - prev_theta) / (prev_theta + 1e-6)
        
        revenue_std = np.std(revenue_hist[-1000:])
        theta_std = np.std([np.mean(hist[-1000:]) for hist in theta_hist if len(hist) > 1000])
    else:
        revenue_change = theta_change = revenue_std = theta_std = None
    
    print(f"Final avg theta: {final_avg_theta:.3f} (theory: {(n_agents-1)/n_agents:.3f})")
    print(f"Final efficiency: {final_efficiency:.3f} (perfect: 1.0)")
    print(f"Final revenue: {final_revenue:.3f}")
    
    if revenue_change is not None:
        print(f"\n=== Convergence Analysis ===")
        print(f"Revenue change (last 1k vs prev 1k): {revenue_change*100:.2f}%")
        print(f"Revenue std (last 1k): {revenue_std:.4f} (lower = more stable)")
        print(f"Theta change (last 1k vs prev 1k): {theta_change*100:.2f}%")
        print(f"Theta std (last 1k): {theta_std:.4f} (lower = more stable)")
        print(f"\nConvergence criteria:")
        print(f"  - Revenue change < 5%: {'✓' if revenue_change < 0.05 else '✗'}")
        print(f"  - Theta change < 5%: {'✓' if theta_change < 0.05 else '✗'}")
        print(f"  - Revenue std < 0.05: {'✓' if revenue_std < 0.05 else '✗'}")
        print(f"  - Theta std < 0.05: {'✓' if theta_std < 0.05 else '✗'}")
    
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
        'agent_type': agent_type,  # Store agent type for strategy analysis
        'pg_outputs': pg_outputs[-1000:] if len(pg_outputs) > 1000 else pg_outputs  # Store last 1000 rounds of P&G outputs
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
    
    elif agent_type == "ppo":
        # PPO strategies: Show theta distribution over time and current values
        fig, axes = plt.subplots(2, (n_agents + 1) // 2, figsize=(15, 10))
        if n_agents == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, agent in enumerate(agents):
            ax = axes[i]
            mean_theta = strategies[i]['mean_theta']
            current_theta = strategies[i]['current_theta']
            
            # Extract theta history if available
            if hasattr(agent, 'history') and len(agent.history) > 0:
                thetas = [h[3] for h in agent.history]  # theta is index 3
                # Plot histogram of theta values
                ax.hist(thetas, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(x=mean_theta, color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_theta:.3f}')
                ax.axvline(x=current_theta, color='g', linestyle='--', linewidth=2,
                          label=f'Current: {current_theta:.3f}')
                ax.axvline(x=(n_agents-1)/n_agents, color='orange', linestyle='--', 
                          label=f'Theory: {(n_agents-1)/n_agents:.3f}')
                ax.set_xlabel('Theta')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Agent {i} Theta Distribution\n(Mean: {mean_theta:.3f}, Current: {current_theta:.3f})')
            else:
                # If no history, just show current theta
                ax.barh(0, current_theta, height=0.5, alpha=0.7, label=f'Current: {current_theta:.3f}')
                ax.axvline(x=(n_agents-1)/n_agents, color='orange', linestyle='--', 
                          label=f'Theory: {(n_agents-1)/n_agents:.3f}')
                ax.set_xlabel('Theta')
                ax.set_ylabel('')
                ax.set_title(f'Agent {i} Current Theta: {current_theta:.3f}')
                ax.set_ylim(-0.5, 0.5)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_agents, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"PPO strategy distributions saved to {save_path}")
        plt.close()

def print_final_pg_strategy(auctioneer, n_agents: int):
    """
    Print the final learned P & G network strategy by testing on sample bids
    """
    print("\nTesting P & G Networks on Sample Bid Scenarios:")
    print("-" * 60)
    
    # Test on a few different bid scenarios
    test_scenarios = [
        ("Uniform low bids", np.array([0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 0.11, 0.13, 0.1, 0.12])),
        ("Uniform high bids", np.array([0.8, 0.82, 0.81, 0.83, 0.8, 0.82, 0.81, 0.83, 0.8, 0.82])),
        ("One high bidder", np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])),
        ("Two high bidders", np.array([0.1, 0.8, 0.1, 0.1, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1])),
    ]
    
    for scenario_name, bids in test_scenarios:
        winner_idx, allocation_probs = auctioneer.G_network.allocate(bids, auctioneer.allocation_temperature)
        payments = auctioneer.P_network.compute_payments(bids, winner_idx, allocation_probs)
        
        print(f"\n{scenario_name}:")
        print(f"  Bids: {bids}")
        print(f"  Winner: Agent {winner_idx} (bid={bids[winner_idx]:.3f})")
        print(f"  Payment: {payments[winner_idx]:.4f} (ratio: {payments[winner_idx]/bids[winner_idx]:.2f}x bid)")
        print(f"  Allocation probs (top 3): {sorted(zip(range(n_agents), allocation_probs), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Compute allocation entropy (measure of diversity)
    # Test on uniform bids to see baseline allocation behavior
    uniform_bids = np.ones(n_agents) * 0.5
    _, allocation_probs = auctioneer.G_network.allocate(uniform_bids, auctioneer.allocation_temperature)
    allocation_entropy = -np.sum(allocation_probs * np.log(allocation_probs + 1e-10))
    max_entropy = np.log(n_agents)
    allocation_diversity = allocation_entropy / max_entropy
    
    print(f"\nAllocation Diversity (on uniform bids): {allocation_diversity:.3f}")
    print(f"  (1.0 = uniform distribution, 0.0 = always same agent)")
    print(f"  Current allocation: {allocation_probs}")

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
    
    elif agent_type == "ppo":
        print("\nPPO Agents:")
        print(f"{'Agent':<10} {'Mean Theta (last 100)':<25} {'Current Theta':<20}")
        print("-" * 80)
        
        for i in range(n_agents):
            mean_theta = strategies[i]['mean_theta']
            current_theta = strategies[i]['current_theta']
            print(f"{i:<10} {mean_theta:<25.3f} {current_theta:<20.3f}")
        
        # Show theta statistics
        print("\n" + "-" * 80)
        print("PPO Strategy Statistics:")
        mean_thetas = [strategies[i]['mean_theta'] for i in range(n_agents)]
        current_thetas = [strategies[i]['current_theta'] for i in range(n_agents)]
        print(f"  Mean theta across agents: {np.mean(mean_thetas):.3f} ± {np.std(mean_thetas):.3f}")
        print(f"  Current theta across agents: {np.mean(current_thetas):.3f} ± {np.std(current_thetas):.3f}")
        print(f"  Theta range: [{np.min(current_thetas):.3f}, {np.max(current_thetas):.3f}]")
    
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
    # Experiment parameters
    training_interval = 200
    lr_auctioneer = 2e-4
    
    print(f"Agents: {n_agents}")
    print(f"Rounds: {n_rounds}")
    print(f"Alternating training: Every {training_interval} rounds")
    print(f"Auctioneer learning rate: {lr_auctioneer}")
    print("="*60)
    
    # Test with PPO agents
    agents, auctioneer, metrics = run_amd_simulation(
        n_agents=n_agents,
        n_rounds=n_rounds,
        info_type=InformationType.FULL_REVELATION,  # PPO can use any info type
        agent_type="ppo",  # Use PPO (much faster than regret matching)
        theta_options=None,  # Not needed for PPO
        alternate_training=True,
        training_interval=training_interval,
        lr_auctioneer=lr_auctioneer
    )
    
    # Create experiment identifier for graph filenames
    exp_id = f"interval{training_interval}_lr{lr_auctioneer:.0e}".replace('.', 'p').replace('+', '')
    
    print("\n" + "=" * 60)
    print("Generating Convergence Plots...")
    print("=" * 60)
    
    # Plot convergence with experiment identifier
    convergence_path = f'graphs/amd_convergence_{exp_id}.png'
    plot_amd_convergence(metrics, n_agents, save_path=convergence_path)
    
    # Extract and visualize strategies
    print("\n" + "=" * 60)
    print("Analyzing Learned Strategies...")
    print("=" * 60)
    
    # Get agent type from metrics or use default
    agent_type_used = metrics.get('agent_type', "ppo")
    
    # Print strategy summary
    print_strategy_summary(agents, agent_type=agent_type_used, n_agents=n_agents)
    
    # Plot strategy visualizations with experiment identifier
    strategies_path = f'graphs/agent_strategies_{exp_id}.png'
    plot_agent_strategies(agents, agent_type=agent_type_used, n_agents=n_agents,
                         save_path=strategies_path)
    
    # Print final P & G network strategy
    print("\n" + "=" * 60)
    print("FINAL P & G NETWORK STRATEGY")
    print("=" * 60)
    print_final_pg_strategy(auctioneer, n_agents)
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print("Final Results:")
    print(f"  Average Theta: {metrics['final_avg_theta']:.3f}")
    print(f"  Final Efficiency: {metrics['final_efficiency']:.3f}")
    print(f"  Final Revenue: {metrics['final_revenue']:.3f}")
    print(f"\nPlots saved:")
    print(f"  Convergence: {convergence_path}")
    print(f"  Strategies: {strategies_path}")

