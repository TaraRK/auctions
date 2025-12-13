import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Set
from matplotlib import pyplot as plt
from scipy.optimize import LinearConstraint, Bounds, milp
from ppo_agent import PPOAgent
from agent import Agent


@dataclass
class BundleBid:
    """Represents a bid on a bundle of items"""
    bundle: Set[int]  # Set of item indices
    bid_amount: float  # Bid amount for this bundle
    value: float  # Agent's private value for this bundle


@dataclass
class CombinatorialAuctionOutcome:
    """Outcome of a combinatorial auction"""
    winning_bids: List[Tuple[int, int]]  # List of (agent_id, bid_index) tuples for winning bids
    item_assignments: np.ndarray  # [n_items] - agent_id who won each item (-1 if not won)
    payments: np.ndarray  # [n_agents] - total payment per agent
    utilities: np.ndarray  # [n_agents] - total utility per agent
    revenue: float  # Total revenue collected


class CombinatorialAuction:
    """
    Multi-item combinatorial sealed-bid auction with XOR semantics.
    Uses Integer Linear Programming (ILP) for winner determination.
    """
    def __init__(self, n_agents: int, n_items: int):
        self.n_agents = n_agents
        self.n_items = n_items
    
    def solve_winner_determination_ilp(
        self, 
        all_bids: List[List[BundleBid]]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Solve winner determination using Integer Linear Programming.
        
        Args:
            all_bids: List of length n_agents, each is a list of BundleBid objects (XOR bids)
            
        Returns:
            winning_bids: List of (agent_id, bid_index) tuples
            item_assignments: [n_items] array with agent_id for each item (-1 if not assigned)
        """
        # Flatten bids: create list of (agent_id, bid_index, bundle, bid_amount)
        bid_list = []
        for agent_id, agent_bids in enumerate(all_bids):
            for bid_idx, bid in enumerate(agent_bids):
                bid_list.append((agent_id, bid_idx, bid.bundle, bid.bid_amount))
        
        n_total_bids = len(bid_list)
        
        if n_total_bids == 0:
            return [], np.full(self.n_items, -1, dtype=int)
        
        # Decision variables: x[i] = 1 if bid i is accepted, 0 otherwise
        # Objective: maximize sum of bid amounts
        c = -np.array([bid[3] for bid in bid_list])  # Negative because we minimize
        
        # Constraints:
        # 1. Each item can be won by at most one agent (across all bundles)
        # 2. Each agent can win at most one bundle (XOR semantics)
        
        # Build constraint matrix
        n_constraints = self.n_items + self.n_agents
        A = np.zeros((n_constraints, n_total_bids))
        b = np.ones(n_constraints)  # At most 1 for each constraint
        
        # Constraint 1: Each item can be in at most one winning bundle
        for item_idx in range(self.n_items):
            for bid_idx, (agent_id, bid_bundle_idx, bundle, _) in enumerate(bid_list):
                if item_idx in bundle:
                    A[item_idx, bid_idx] = 1
        
        # Constraint 2: Each agent can win at most one bundle (XOR)
        for agent_id in range(self.n_agents):
            for bid_idx, (bid_agent_id, _, _, _) in enumerate(bid_list):
                if bid_agent_id == agent_id:
                    A[self.n_items + agent_id, bid_idx] = 1
        
        # Bounds: binary variables
        bounds = Bounds(0, 1)
        integrality = np.ones(n_total_bids, dtype=int)  # All variables are integers
        
        # Solve ILP
        try:
            result = milp(
                c=c,
                constraints=LinearConstraint(A, ub=b),
                bounds=bounds,
                integrality=integrality,
                options={'disp': False}
            )
            
            if result.success and result.x is not None:
                solution = result.x
                winning_bids = []
                item_assignments = np.full(self.n_items, -1, dtype=int)
                
                # Extract winning bids
                for bid_idx, (agent_id, bid_bundle_idx, bundle, _) in enumerate(bid_list):
                    if solution[bid_idx] > 0.5:  # Binary variable is 1
                        winning_bids.append((agent_id, bid_bundle_idx))
                        # Assign items to this agent
                        for item_idx in bundle:
                            if item_assignments[item_idx] == -1:
                                item_assignments[item_idx] = agent_id
                
                return winning_bids, item_assignments
            else:
                # ILP failed, use greedy
                return self._greedy_allocation(bid_list)
                
        except Exception as e:
            # Fallback to greedy if ILP solver fails
            print(f"ILP solver error: {e}, using greedy allocation")
            return self._greedy_allocation(bid_list)
    
    def _greedy_allocation(self, bid_list: List[Tuple]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Greedy fallback: select bids in order of bid amount (descending)"""
        # Sort by bid amount (descending)
        sorted_bids = sorted(bid_list, key=lambda x: x[3], reverse=True)
        
        winning_bids = []
        item_assignments = np.full(self.n_items, -1, dtype=int)
        used_agents = set()
        used_items = set()
        
        for agent_id, bid_idx, bundle, bid_amount in sorted_bids:
            # Check XOR constraint: agent not already winning
            if agent_id in used_agents:
                continue
            
            # Check item constraints: no item already assigned
            if any(item in used_items for item in bundle):
                continue
            
            # Accept this bid
            winning_bids.append((agent_id, bid_idx))
            used_agents.add(agent_id)
            for item in bundle:
                used_items.add(item)
                item_assignments[item] = agent_id
        
        return winning_bids, item_assignments
    
    def run_auction(
        self, 
        all_bids: List[List[BundleBid]]
    ) -> CombinatorialAuctionOutcome:
        """
        Run combinatorial auction with XOR semantics.
        
        Args:
            all_bids: List of length n_agents, each is a list of BundleBid objects
            
        Returns:
            CombinatorialAuctionOutcome
        """
        # Solve winner determination
        winning_bids, item_assignments = self.solve_winner_determination_ilp(all_bids)
        
        # Calculate payments (first-price: winners pay their bid)
        payments = np.zeros(self.n_agents)
        utilities = np.zeros(self.n_agents)
        revenue = 0.0
        
        for agent_id, bid_idx in winning_bids:
            bid = all_bids[agent_id][bid_idx]
            payments[agent_id] = bid.bid_amount
            utilities[agent_id] = bid.value - bid.bid_amount
            revenue += bid.bid_amount
        
        return CombinatorialAuctionOutcome(
            winning_bids=winning_bids,
            item_assignments=item_assignments,
            payments=payments,
            utilities=utilities,
            revenue=revenue,
        )


class CombinatorialPPOAgent(PPOAgent):
    """
    PPO agent adapted for combinatorial auctions.
    Can submit multiple bundle bids (XOR semantics).
    """
    def __init__(
        self,
        agent_id: int,
        n_items: int,
        max_bundles: int = 5,  # Maximum number of bundle bids per agent
        initial_budget: float = 100.0,
        total_auctions: int = 50,
        **ppo_kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            initial_budget=initial_budget,
            total_auctions=total_auctions,
            **ppo_kwargs
        )
        self.n_items = n_items
        self.max_bundles = max_bundles
        
        # Extended state: [value, budget_fraction, auctions_remaining_fraction, n_items]
        # For combinatorial, we need to handle multiple bundles
        # We'll use a simplified approach: generate bundles and bid amounts
        
        # Actor: outputs bid amount as fraction of value for each potential bundle
        # Simplified: we'll generate a fixed set of bundle types and bid on them
        self.bundle_types = self._generate_bundle_types()
        
        # Actor network: state -> bid amounts for each bundle type
        # State: [value, budget_fraction, auctions_remaining_fraction]
        # Output: [max_bundles] bid amounts (as fraction of value)
        self.actor_mean = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.max_bundles),
            nn.Sigmoid()  # Bid as fraction of value [0, 1]
        )
        
        # Initialize log_std to encourage more exploration initially
        self.actor_log_std = nn.Parameter(torch.ones(self.max_bundles) * 0.5)  # Start with higher std for exploration
        
        # Critic remains the same
        self.critic = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.actor_mean.parameters())
            + list(self.critic.parameters())
            + [self.actor_log_std],
            lr=self.lr,
        )
        
        # Current bundle values and bids
        self.current_bundle_values = None
        self.current_bundle_bids = None
        self.current_bundle_logprobs = None
    
    def reset(self):
        """Reset agent state for new episode"""
        super().reset()  # Reset PPO agent (budget, auctions_remaining, etc.)
        # Reset combinatorial-specific state
        self.current_bundle_values = None
        self.current_bundle_bids = None
    
    def _generate_bundle_types(self) -> List[Set[int]]:
        """Generate a set of bundle types to bid on"""
        bundles = []
        # Single items
        for i in range(self.n_items):
            bundles.append({i})
        # Pairs (if n_items > 1)
        if self.n_items > 1:
            for i in range(self.n_items):
                for j in range(i + 1, min(i + 2, self.n_items)):  # Limit to adjacent pairs
                    bundles.append({i, j})
        # All items bundle
        if self.n_items > 0:
            bundles.append(set(range(self.n_items)))
        
        # Limit to max_bundles
        return bundles[:self.max_bundles]
    
    def draw_bundle_values(self) -> List[float]:
        """
        Draw private values for each bundle type.
        For simplicity, we use additive values with some complementarity.
        """
        # Base value for each item
        item_values = np.random.uniform(0.5, 1.0, size=self.n_items)
        
        bundle_values = []
        for bundle in self.bundle_types:
            # Base value: sum of item values
            base_value = sum(item_values[i] for i in bundle)
            # Add complementarity bonus (small random factor)
            complementarity = 1.0 + np.random.uniform(0.0, 0.2) * len(bundle)
            bundle_values.append(base_value * complementarity)
        
        return bundle_values
    
    def choose_bundle_bids(self) -> List[BundleBid]:
        """
        Choose bundle bids using PPO policy.
        Returns list of BundleBid objects.
        """
        if self.current_state is None:
            # Use average bundle value as state representation
            bundle_values = self.draw_bundle_values()
            avg_value = np.mean(bundle_values)
            self.current_value = avg_value
            self.current_state = self.get_state(avg_value)
            self.current_bundle_values = bundle_values
        
        with torch.no_grad():
            s = self.current_state.unsqueeze(0)  # [1, 3]
            
            # Get mean bid fractions
            mean_bid_fractions = self.actor_mean(s)  # [1, max_bundles]
            log_std = self.actor_log_std.clamp(min=-5.0, max=2.0)
            std = log_std.exp().unsqueeze(0)
            
            # Sample bid fractions
            dist = torch.distributions.Normal(mean_bid_fractions, std)
            bid_fractions = dist.sample()  # [1, max_bundles]
            bid_fractions = torch.clamp(bid_fractions, 0.0, 1.0)
            logprobs = dist.log_prob(bid_fractions).sum(dim=1)  # [1]
            
            bid_fractions = bid_fractions.squeeze(0).cpu().numpy()
            logprob = logprobs.item()
        
        # Create bundle bids
        bundle_bids = []
        for i, bundle in enumerate(self.bundle_types):
            value = self.current_bundle_values[i]
            bid_amount = value * bid_fractions[i]
            
            # Only include bids above a threshold
            if bid_amount > 0.01:  # Minimum bid threshold
                bundle_bids.append(BundleBid(
                    bundle=bundle,
                    bid_amount=bid_amount,
                    value=value
                ))
        
        self.current_bundle_bids = bundle_bids
        self.current_bundle_logprobs = logprob
        
        return bundle_bids
    
    def update(self, bundle_bids: List[BundleBid], outcome: CombinatorialAuctionOutcome):
        """
        Update PPO agent based on auction outcome.
        """
        # Check if we won any bundle
        won = False
        winning_bundle_value = 0.0
        payment = 0.0
        
        for agent_id, bid_idx in outcome.winning_bids:
            if agent_id == self.agent_id:
                won = True
                winning_bundle = bundle_bids[bid_idx]
                winning_bundle_value = winning_bundle.value
                payment = outcome.payments[self.agent_id]
                break
        
        utility = outcome.utilities[self.agent_id]
        
        # Budget constraint
        if won and payment > self.budget + 1e-12:
            utility = 0.0
            payment = 0.0
            won = False
        
        # IMPORTANT: If utility is exactly 0 or negative for a winner, this means
        # the agent bid >= their value, which shouldn't happen with proper clamping.
        # But if it does, we should still give a small negative reward to discourage this.
        if won and utility <= 0:
            # Agent won but bid too high (at or above value) - penalize this
            utility = -0.1  # Stronger penalty for winning with no profit
        
        # Reward shaping: encourage profitable wins by giving bonus for positive utility
        # This helps agents learn to shade their bids appropriately
        if won and utility > 0:
            # Small bonus for making profit (encourages shading)
            utility = utility * 1.1  # 10% bonus for profitable wins
        
        # Reward shaping: add small negative reward for losing to encourage learning
        # This helps with sparse reward problem
        if not won:
            utility = -0.01  # Small negative reward for losing (instead of 0)
        
        # Scale rewards to help with learning (optional - can help with gradient stability)
        # utility = utility / 10.0  # Uncomment if utilities are too large
        
        self.budget -= payment
        self.auctions_remaining -= 1
        done = (self.budget <= 0.0) or (self.auctions_remaining == 0)
        
        # Track history
        total_bid = sum(bid.bid_amount for bid in bundle_bids)
        self.history.append((winning_bundle_value, total_bid, utility, 0.0))  # theta not used here
        
        # Store transition for PPO
        with torch.no_grad():
            val = self.critic(self.current_state.unsqueeze(0)).squeeze().item()
        
        # Store bid fractions for all bundles as action vector
        bid_fractions = np.zeros(self.max_bundles)
        if bundle_bids:
            for i, bundle in enumerate(self.bundle_types):
                # Find matching bid
                for bid in bundle_bids:
                    if bid.bundle == bundle:
                        bid_fractions[i] = bid.bid_amount / bid.value
                        break
        
        self.buffer["states"].append(self.current_state.numpy())
        self.buffer["actions"].append(bid_fractions)  # Store full action vector
        self.buffer["logprobs"].append(self.current_bundle_logprobs)
        self.buffer["rewards"].append(utility)
        self.buffer["values"].append(val)
        self.buffer["dones"].append(float(done))
        
        # Update when buffer full or episode done
        if len(self.buffer["states"]) >= self.buffer_size or done:
            self._ppo_update_combinatorial()
            for k in self.buffer:
                self.buffer[k] = []
        
        # Reset for next round
        self.current_bundle_values = None
        self.current_bundle_bids = None
    
    def _ppo_update_combinatorial(self):
        """PPO update for combinatorial action space"""
        states = torch.tensor(np.array(self.buffer["states"]), dtype=torch.float32)  # [T, 3]
        actions = torch.tensor(np.array(self.buffer["actions"]), dtype=torch.float32)  # [T, max_bundles]
        old_logprobs = torch.tensor(self.buffer["logprobs"], dtype=torch.float32)  # [T]
        rewards = torch.tensor(self.buffer["rewards"], dtype=torch.float32)  # [T]
        dones = torch.tensor(self.buffer["dones"], dtype=torch.float32)  # [T]
        
        # Critic values at states (fixed for GAE computation)
        with torch.no_grad():
            values = self.critic(states).squeeze()  # [T]
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_nonterminal = 1.0 - dones[t]
                    next_value = 0.0 if dones[t] > 0.5 else values[t]
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.update_epochs):
            # Get policy distribution
            s = states  # [T, 3]
            mean_bid_fractions = self.actor_mean(s)  # [T, max_bundles]
            log_std = self.actor_log_std.clamp(min=-5.0, max=2.0)
            std = log_std.exp().unsqueeze(0).expand_as(mean_bid_fractions)
            
            dist = torch.distributions.Normal(mean_bid_fractions, std)
            new_logprobs = dist.log_prob(actions).sum(dim=1)  # [T]
            entropy = dist.entropy().sum(dim=1).mean()
            
            new_values = self.critic(states).squeeze()
            
            ratio = (new_logprobs - old_logprobs).exp()
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * ratio.clamp(1.0 - self.clip_coef, 1.0 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()
            
            loss = pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_mean.parameters()) + list(self.critic.parameters()) + [self.actor_log_std],
                self.max_grad_norm,
            )
            self.optimizer.step()


def plot_combinatorial_results(
    n_agents, n_items, bid_fraction_hist, avg_bid_fraction_hist, 
    revenue_hist, utility_hist, win_rate_hist, bundle_usage_hist
):
    """Plot combinatorial auction simulation results"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    rounds = np.arange(len(avg_bid_fraction_hist))
    window = min(100, len(avg_bid_fraction_hist) // 10) if len(avg_bid_fraction_hist) > 10 else 1
    
    # ----- 1. Bid fraction convergence -----
    ax = axes[0, 0]
    if len(avg_bid_fraction_hist) >= window:
        bid_smooth = np.convolve(avg_bid_fraction_hist, np.ones(window) / window, mode='valid')
        ax.plot(bid_smooth, label='avg bid fraction', linewidth=2)
    else:
        ax.plot(avg_bid_fraction_hist, label='avg bid fraction', linewidth=2)
    
    # Show individual agents' bid fractions (last 1000 rounds)
    if len(bid_fraction_hist) > 0 and len(bid_fraction_hist[0]) > 0:
        bid_arr = np.array(bid_fraction_hist)  # [n_agents, T]
        if bid_arr.shape[1] > 100:
            # Show mean ± std
            mean_bid = bid_arr.mean(axis=0)
            std_bid = bid_arr.std(axis=0)
            if len(mean_bid) >= window:
                mean_smooth = np.convolve(mean_bid, np.ones(window) / window, mode='valid')
                std_smooth = np.convolve(std_bid, np.ones(window) / window, mode='valid')
                ax.fill_between(range(len(mean_smooth)), 
                               mean_smooth - std_smooth, 
                               mean_smooth + std_smooth,
                               alpha=0.2, label='±1 std')
    
    ax.set_xlabel('round')
    ax.set_ylabel('bid fraction (bid/value)')
    ax.set_title('Bid Fraction Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ----- 2. Revenue over time -----
    ax = axes[0, 1]
    if len(revenue_hist) >= window:
        revenue_smooth = np.convolve(revenue_hist, np.ones(window) / window, mode='valid')
        ax.plot(revenue_smooth, linewidth=2)
    else:
        ax.plot(revenue_hist, linewidth=2)
    ax.set_xlabel('round')
    ax.set_ylabel('total revenue')
    ax.set_title('Auctioneer Revenue Over Time')
    ax.grid(True, alpha=0.3)
    
    # ----- 3. Average utility per agent -----
    ax = axes[1, 0]
    if len(utility_hist) > 0:
        utility_arr = np.array(utility_hist)  # [T, n_agents]
        mean_utility = utility_arr.mean(axis=1)
        if len(mean_utility) >= window:
            utility_smooth = np.convolve(mean_utility, np.ones(window) / window, mode='valid')
            ax.plot(utility_smooth, label='mean utility', linewidth=2)
        else:
            ax.plot(mean_utility, label='mean utility', linewidth=2)
        
        # Show std
        std_utility = utility_arr.std(axis=1)
        if len(std_utility) >= window:
            std_smooth = np.convolve(std_utility, np.ones(window) / window, mode='valid')
            mean_smooth = np.convolve(mean_utility, np.ones(window) / window, mode='valid')
            ax.fill_between(range(len(mean_smooth)), 
                           mean_smooth - std_smooth, 
                           mean_smooth + std_smooth,
                           alpha=0.2, label='±1 std')
    ax.set_xlabel('round')
    ax.set_ylabel('average utility')
    ax.set_title('Agent Utility Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ----- 4. Win rate per agent -----
    ax = axes[1, 1]
    if len(win_rate_hist) > 0:
        win_arr = np.array(win_rate_hist)  # [T, n_agents]
        # Average win rate over last portion
        tail = min(500, win_arr.shape[0])
        final_win_rates = win_arr[-tail:].mean(axis=0)
        ax.bar(range(n_agents), final_win_rates, alpha=0.7)
        ax.axhline(y=1.0/n_agents, color='r', linestyle='--', 
                   label=f'equal share (1/n={1.0/n_agents:.3f})')
    ax.set_xlabel('agent ID')
    ax.set_ylabel('win rate (last 500 rounds)')
    ax.set_title('Win Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ----- 5. Bundle usage over time -----
    ax = axes[2, 0]
    if len(bundle_usage_hist) > 0:
        # bundle_usage_hist is list of dicts: [{bundle_str: count}, ...]
        # Aggregate over time
        all_bundles = set()
        for usage in bundle_usage_hist:
            all_bundles.update(usage.keys())
        
        if len(all_bundles) > 0:
            bundle_list = sorted(all_bundles)
            usage_matrix = np.zeros((len(bundle_usage_hist), len(bundle_list)))
            for t, usage in enumerate(bundle_usage_hist):
                for i, bundle in enumerate(bundle_list):
                    usage_matrix[t, i] = usage.get(bundle, 0)
            
            # Plot top 5 most used bundles
            total_usage = usage_matrix.sum(axis=0)
            top_indices = np.argsort(total_usage)[-5:][::-1]
            
            for idx in top_indices:
                bundle_str = bundle_list[idx]
                if len(usage_matrix[:, idx]) >= window:
                    smooth = np.convolve(usage_matrix[:, idx], np.ones(window) / window, mode='valid')
                    ax.plot(smooth, label=f'bundle {bundle_str}', linewidth=1.5)
                else:
                    ax.plot(usage_matrix[:, idx], label=f'bundle {bundle_str}', linewidth=1.5)
    
    ax.set_xlabel('round')
    ax.set_ylabel('usage count (rolling avg)')
    ax.set_title('Bundle Usage Over Time (Top 5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ----- 6. Final bid fraction distribution -----
    ax = axes[2, 1]
    if len(bid_fraction_hist) > 0:
        bid_arr = np.array(bid_fraction_hist)  # [n_agents, T]
        tail = min(500, bid_arr.shape[1])
        final_bid_fractions = bid_arr[:, -tail:].mean(axis=1)
        ax.hist(final_bid_fractions, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(final_bid_fractions), color='r', linestyle='--', 
                   label=f'mean={np.mean(final_bid_fractions):.3f}')
    ax.set_xlabel('avg bid fraction (last 500 rounds)')
    ax.set_ylabel('count')
    ax.set_title('Final Bid Fraction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('combinatorial_auction_learning.png', dpi=150)
    plt.show()
    
    # Console summary
    print(f"\nFinal results (n_agents={n_agents}, n_items={n_items}):")
    if len(avg_bid_fraction_hist) > 0:
        print(f"Final avg bid fraction (last 100 rounds): {np.mean(avg_bid_fraction_hist[-100:]):.3f}")
    if len(revenue_hist) > 0:
        print(f"Final avg revenue (last 100 rounds): {np.mean(revenue_hist[-100:]):.3f}")
    if len(utility_hist) > 0:
        utility_arr = np.array(utility_hist)
        print(f"Final avg utility (last 100 rounds): {utility_arr[-100:].mean():.3f}")
        print(f"Utility trend: first 100={utility_arr[:100].mean():.3f}, last 100={utility_arr[-100:].mean():.3f}")
        print(f"Positive utility rate (last 100): {(utility_arr[-100:] > 0).sum() / utility_arr[-100:].size * 100:.1f}%")
    if len(win_rate_hist) > 0:
        win_arr = np.array(win_rate_hist)
        tail = min(500, win_arr.shape[0])
        final_win_rates = win_arr[-tail:].mean(axis=0)
        print(f"Win rate std (last {tail} rounds): {final_win_rates.std():.3f} (lower = more equal)")
        print(f"Overall win rate: {win_arr.mean() * 100:.1f}%")


def run_combinatorial_simulation(n_agents: int, n_items: int, n_rounds: int, auctions_per_episode: int = 50):
    """
    Run simulation of combinatorial auctions with PPO agents.
    
    Args:
        n_agents: Number of agents
        n_items: Number of items
        n_rounds: Number of episodes
        auctions_per_episode: Number of auctions per episode (for PPO agents)
    """
    auction = CombinatorialAuction(n_agents, n_items)
    agents = [
        CombinatorialPPOAgent(
            i, 
            n_items, 
            max_bundles=min(5, 2**n_items),
            initial_budget=100.0,
            total_auctions=auctions_per_episode
        )
        for i in range(n_agents)
    ]
    
    # Tracking
    total_rounds = 0
    bid_fraction_hist = [[] for _ in range(n_agents)]  # Per-agent bid fractions
    avg_bid_fraction_hist = []
    revenue_hist = []
    utility_hist = []  # [T, n_agents]
    win_rate_hist = []  # [T, n_agents] - binary win/loss per round
    bundle_usage_hist = []  # List of dicts: [{bundle_str: count}, ...]
    
    # Track wins per agent
    total_wins = np.zeros(n_agents)
    
    for episode in range(n_rounds):
        # Reset all agents for new episode
        for agent in agents:
            agent.reset()
        
        for round_idx in range(auctions_per_episode):
            # Each agent generates bundle bids
            all_bids = []
            bundle_usage = {}
            
            for agent in agents:
                # Draw values and generate bids
                agent.current_bundle_values = agent.draw_bundle_values()
                avg_value = np.mean(agent.current_bundle_values)
                agent.current_value = avg_value
                agent.current_state = agent.get_state(avg_value)
                
                bundle_bids = agent.choose_bundle_bids()
                all_bids.append(bundle_bids)
                
                # Track bundle usage
                for bid in bundle_bids:
                    bundle_str = str(sorted(bid.bundle))
                    bundle_usage[bundle_str] = bundle_usage.get(bundle_str, 0) + 1
            
            bundle_usage_hist.append(bundle_usage)
            
            # Run auction
            outcome = auction.run_auction(all_bids)
            
            # Track wins
            round_wins = np.zeros(n_agents)
            for agent_id, bid_idx in outcome.winning_bids:
                round_wins[agent_id] = 1
                total_wins[agent_id] += 1
            win_rate_hist.append(round_wins)
            
            # Agents update
            round_utilities = np.zeros(n_agents)
            for i, agent in enumerate(agents):
                agent.update(all_bids[i], outcome)
                # Use utility from outcome (before reward shaping modifications)
                round_utilities[i] = outcome.utilities[i]
                
                # Track bid fractions per agent
                if all_bids[i]:
                    avg_bid_frac = np.mean([bid.bid_amount / bid.value for bid in all_bids[i]])
                    bid_fraction_hist[i].append(avg_bid_frac)
                else:
                    bid_fraction_hist[i].append(0.0)
            
            # Diagnostic: check if winning agents have positive utility
            winning_agent_ids = [agent_id for agent_id, _ in outcome.winning_bids]
            if len(winning_agent_ids) > 0:
                winning_utilities = [outcome.utilities[aid] for aid in winning_agent_ids]
                if total_rounds % 100 == 0:
                    print(f"  [DEBUG] Winning agents utilities: min={min(winning_utilities):.4f}, "
                          f"max={max(winning_utilities):.4f}, mean={np.mean(winning_utilities):.4f}")
                    # Check bid fractions for winning agents
                    for aid in winning_agent_ids[:3]:  # Show first 3 winners
                        if all_bids[aid]:
                            winning_bid_idx = next(bid_idx for agent_id, bid_idx in outcome.winning_bids if agent_id == aid)
                            winning_bid = all_bids[aid][winning_bid_idx]
                            bid_frac = winning_bid.bid_amount / winning_bid.value
                            print(f"    Agent {aid}: value={winning_bid.value:.3f}, bid={winning_bid.bid_amount:.3f}, "
                                  f"bid_frac={bid_frac:.3f}, utility={outcome.utilities[aid]:.3f}")
            
            utility_hist.append(round_utilities)
            
            # Tracking aggregates
            revenue_hist.append(outcome.revenue)
            
            # Average bid fraction across all agents
            avg_bid_fraction = np.mean([
                np.mean([bid.bid_amount / bid.value for bid in bids]) 
                for bids in all_bids if bids
            ]) if any(all_bids) else 0.0
            avg_bid_fraction_hist.append(avg_bid_fraction)
            
            total_rounds += 1
            
            if total_rounds % 100 == 0:
                win_rate = (round_wins.sum() / n_agents) * 100
                positive_utility_rate = (round_utilities > 0).sum() / n_agents * 100
                zero_utility_rate = (round_utilities == 0).sum() / n_agents * 100
                negative_utility_rate = (round_utilities < 0).sum() / n_agents * 100
                avg_budget = np.mean([agent.budget for agent in agents])
                
                # Check utilities for winning agents specifically
                winning_agent_ids = [agent_id for agent_id, _ in outcome.winning_bids]
                if len(winning_agent_ids) > 0:
                    winning_utilities = [round_utilities[aid] for aid in winning_agent_ids]
                    winning_positive_rate = sum(1 for u in winning_utilities if u > 0) / len(winning_utilities) * 100
                else:
                    winning_positive_rate = 0.0
                
                print(f"Round {total_rounds}: revenue={outcome.revenue:.3f}, "
                      f"avg_bid_fraction={avg_bid_fraction:.3f}, "
                      f"n_winning_bids={len(outcome.winning_bids)}, "
                      f"avg_utility={round_utilities.mean():.3f}, "
                      f"win_rate={win_rate:.1f}%, "
                      f"positive_utility_rate={positive_utility_rate:.1f}%, "
                      f"zero_utility_rate={zero_utility_rate:.1f}%, "
                      f"negative_utility_rate={negative_utility_rate:.1f}%, "
                      f"winning_positive_rate={winning_positive_rate:.1f}%, "
                      f"avg_budget={avg_budget:.2f}")
                
                # Diagnostic: show utility distribution for winners
                if len(winning_agent_ids) > 0 and total_rounds % 500 == 0:
                    print(f"  [DIAG] Utility stats for winners: min={min(winning_utilities):.4f}, "
                          f"max={max(winning_utilities):.4f}, mean={np.mean(winning_utilities):.4f}, "
                          f"median={np.median(winning_utilities):.4f}")
                    # Show a sample of winning bids
                    for aid in winning_agent_ids[:2]:
                        if all_bids[aid]:
                            winning_bid_idx = next(bid_idx for agent_id, bid_idx in outcome.winning_bids if agent_id == aid)
                            winning_bid = all_bids[aid][winning_bid_idx]
                            bid_frac = winning_bid.bid_amount / winning_bid.value
                            util = round_utilities[aid]
                            print(f"    Agent {aid}: value={winning_bid.value:.3f}, bid={winning_bid.bid_amount:.3f}, "
                                  f"bid_frac={bid_frac:.4f}, utility={util:.4f}")
    
    # Plot results
    plot_combinatorial_results(
        n_agents, n_items, bid_fraction_hist, avg_bid_fraction_hist,
        revenue_hist, utility_hist, win_rate_hist, bundle_usage_hist
    )
    
    return agents, utility_hist, revenue_hist, avg_bid_fraction_hist


if __name__ == "__main__":
    # Example usage
    agents, efficiency, revenue, bid_fraction = run_combinatorial_simulation(
        n_agents=10,
        n_items=5,
        n_rounds=1000,
        auctions_per_episode=50
    )

