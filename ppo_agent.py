import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform
from agent import Agent


class PPOAgent(Agent):
    def __init__(
        self,
        agent_id: int,
        initial_budget: float = 100.0,
        total_auctions: int = 50,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        update_epochs: int = 4,
        buffer_size: int = 128,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(agent_id)
        self.initial_budget = initial_budget
        self.total_auctions = total_auctions
        self.budget = initial_budget
        self.auctions_remaining = total_auctions

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.buffer_size = buffer_size
        self.max_grad_norm = max_grad_norm

        # State size is dynamic based on information revelation
        # Base: 3 (value, budget_frac, auctions_frac)
        # + 1 (won indicator)
        # + 1 (winning_bid if revealed)
        # + 10 (losing_bids if revealed, max 10)
        # + 10 (all_bids if FULL_TRANSPARENCY/REVELATION, max 10)
        # Max state size: 3 + 1 + 1 + 10 + 10 = 25
        # We'll use max size to allow flexibility, but state can be smaller
        self.state_dim = 25  # Maximum state dimension
        self.actor_mean = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # no sigmoid; we handle bounds via transforms
        )
        # global log std works; you can later make it state-dependent
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(
            list(self.actor_mean.parameters())
            + list(self.critic.parameters())
            + [self.actor_log_std],
            lr=lr,
        )

        self.buffer = {
            "values": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "dones": [],
            "states": [],
        }

        self.current_value = None
        self.current_state = None
        self.current_theta = None
        self.current_logprob = None

    # ---------- basic episode plumbing ----------

    def reset(self):
        self.budget = self.initial_budget
        self.auctions_remaining = self.total_auctions

    def get_state(self, value: float, agent_info=None) -> torch.Tensor:
        """
        Get state representation including revealed information
        
        Args:
            value: Agent's private value
            agent_info: AgentInformation object with revealed information (optional)
        
        Returns:
            State tensor with:
            - Basic: [value, budget_frac, auctions_frac]
            - Won: [1 if won, 0 if lost]
            - Winning bid: [winning_bid] if revealed
            - Losing bids: [losing_bids...] if revealed (for winner)
            - All bids: [all_bids...] if FULL_TRANSPARENCY or FULL_REVELATION
            - Payment: [own_payment] if FULL_REVELATION
        """
        # Safe division with checks
        budget_frac = self.budget / self.initial_budget if self.initial_budget > 0 else 0.0
        auctions_frac = self.auctions_remaining / self.total_auctions if self.total_auctions > 0 else 0.0
        
        # Ensure finite values
        budget_frac = 0.0 if not np.isfinite(budget_frac) else budget_frac
        auctions_frac = 0.0 if not np.isfinite(auctions_frac) else auctions_frac
        
        state = [
            float(value) if np.isfinite(value) else 0.0,
            float(budget_frac),
            float(auctions_frac),
        ]
        
        if agent_info is not None:
            # Won/lost indicator
            state.append(1.0 if agent_info.won else 0.0)
            
            # Winning bid if revealed (check for NaN/inf)
            if agent_info.winning_bid is not None:
                winning_bid = float(agent_info.winning_bid)
                state.append(winning_bid if np.isfinite(winning_bid) else 0.0)
            
            # Losing bids if revealed (for winner)
            if agent_info.losing_bids is not None:
                # Normalize losing bids (max 10 to keep state size reasonable)
                losing_bids = agent_info.losing_bids[:10]
                # Convert to list and check for NaN/inf
                losing_bids_list = [float(x) if np.isfinite(x) else 0.0 for x in losing_bids.tolist()]
                state.extend(losing_bids_list)
                # Pad with zeros if fewer than 10
                if len(losing_bids_list) < 10:
                    state.extend([0.0] * (10 - len(losing_bids_list)))
            
            # All bids if FULL_TRANSPARENCY or FULL_REVELATION
            if agent_info.all_bids is not None:
                # Normalize all bids (max 10 to keep state size reasonable)
                all_bids = agent_info.all_bids[:10]
                # Convert to list and check for NaN/inf
                all_bids_list = [float(x) if np.isfinite(x) else 0.0 for x in all_bids.tolist()]
                state.extend(all_bids_list)
                # Pad with zeros if fewer than 10
                if len(all_bids_list) < 10:
                    state.extend([0.0] * (10 - len(all_bids_list)))
        
        return torch.tensor(state, dtype=torch.float32)

    def draw_value(self) -> float:
        v = np.random.uniform(0.0, 1.0)
        self.current_value = v
        # Initialize state with just basic info (no agent_info yet)
        basic_state = torch.tensor([
            v,
            self.budget / self.initial_budget,
            self.auctions_remaining / self.total_auctions,
        ], dtype=torch.float32)
        # Pad to max dimension
        self.current_state = torch.zeros(self.state_dim)
        self.current_state[:len(basic_state)] = basic_state
        return v

    # ---------- key fix: consistent, bounded action distribution ----------

    def _max_theta_from_state(self, states: torch.Tensor) -> torch.Tensor:
        """
        per-state feasible upper bound for theta (bid/value).
        theta ∈ [0, min(value, budget)/value] = [0, max_theta]
        """
        # states[..., 0] = value, states[..., 1] = budget_frac
        value = states[..., 0].clamp(min=1e-8)
        budget = states[..., 1] * self.initial_budget
        max_bid = torch.minimum(value, budget)
        max_theta = (max_bid / value).clamp(0.0, 1.0)
        # tiny floor prevents degenerate transforms when budget ≈ 0
        return torch.maximum(max_theta, torch.tensor(1e-6, dtype=states.dtype))

    def _policy_dist(self, states: torch.Tensor):
        """
        base gaussian on R → tanh to (−1,1) → affine to (0,1) → scale to (0, max_theta).
        TransformedDistribution takes care of log-det jacobians, so log_prob is correct.
        """
        # Check for NaN/inf in states before passing to network
        if torch.isnan(states).any() or torch.isinf(states).any():
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        
        loc = self.actor_mean(states)                      # shape [..., 1]
        
        # Check and fix actor_log_std if it's NaN (must happen before using it)
        if torch.isnan(self.actor_log_std).any() or torch.isinf(self.actor_log_std).any():
            with torch.no_grad():
                self.actor_log_std.data.fill_(0.0)  # Reset to initial value
        
        # Check for NaN in loc and replace with 0
        if torch.isnan(loc).any() or torch.isinf(loc).any():
            loc = torch.nan_to_num(loc, nan=0.0, posinf=0.0, neginf=0.0)
        
        # clamp log_std to sane range to avoid pathological std
        # First ensure actor_log_std itself is valid
        if torch.isnan(self.actor_log_std).any() or torch.isinf(self.actor_log_std).any():
            with torch.no_grad():
                self.actor_log_std.data.fill_(0.0)
        
        log_std = self.actor_log_std.clamp(min=-5.0, max=2.0)
        
        # Check log_std after clamping
        if torch.isnan(log_std).any() or torch.isinf(log_std).any():
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=2.0, neginf=-5.0)
        
        std = log_std.exp().expand_as(loc)
        
        # Ensure std is valid (check again after exp)
        if torch.isnan(std).any() or torch.isinf(std).any():
            std = torch.nan_to_num(std, nan=1e-6, posinf=10.0, neginf=1e-6)
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        # Final check before creating distribution
        if torch.isnan(std).any() or torch.isinf(std).any():
            std = torch.full_like(std, 1e-6)  # Fallback to minimum valid std
        
        base = Normal(loc, std)

        max_theta = self._max_theta_from_state(states).unsqueeze(-1)
        transforms = [
            TanhTransform(cache_size=1),                   # (−1,1)
            AffineTransform(loc=0.5, scale=0.5),          # (x+1)/2 → (0,1)
            AffineTransform(loc=0.0, scale=max_theta),    # (0,1) → (0, max_theta)
        ]
        dist = TransformedDistribution(base, transforms)
        return dist, max_theta

    def choose_theta(self) -> float:
        with torch.no_grad():
            s = self.current_state.unsqueeze(0)            # [1,3]
            dist, _ = self._policy_dist(s)
            action = dist.sample()                         # [1,1] in [0, max_theta]
            logprob = dist.log_prob(action).sum(dim=-1)    # [1]
        self.current_theta = action.item()
        self.current_logprob = logprob.item()
        self.theta = self.current_theta
        return self.current_theta

    # ---------- env update + buffer ----------

    def update(self, value: float, chosen_theta: float, outcome):
        bid = value * chosen_theta
        won = (outcome.winner_idx == self.agent_id)
        too_expensive = False
        
        # Get payment: if won, winning_bid is the payment (set in AMD simulation)
        # If lost, payment is 0
        if won:
            price_paid = outcome.winning_bid  # In AMD, winning_bid is set to the actual payment
        else:
            price_paid = 0.0

        # can't pay more than budget
        if won and price_paid > self.budget + 1e-12:
            too_expensive = True
            won = False
            price_paid = 0.0

        # Use utility from outcome (already computed as value - payment)
        utility = outcome.utilities[self.agent_id] if too_expensive is False else 0.0

        # budget & time
        self.budget -= price_paid
        self.auctions_remaining -= 1
        done = (self.budget <= 0.0) or (self.auctions_remaining == 0)

        # track (value, bid, utility, theta)
        self.history.append((value, bid, utility, chosen_theta))

        # Get agent_info from outcome if available (for state representation)
        agent_info = None
        if hasattr(outcome, 'agent_info') and self.agent_id in outcome.agent_info:
            agent_info = outcome.agent_info[self.agent_id]
        
        # Update state with revealed information
        self.current_state = self.get_state(value, agent_info)
        
        # Pad state to max dimension if needed
        state_padded = torch.zeros(self.state_dim)
        state_len = self.current_state.shape[0]
        state_padded[:state_len] = self.current_state
        self.current_state = state_padded

        with torch.no_grad():
            val = self.critic(self.current_state.unsqueeze(0)).squeeze().item()

        # store transition (use tolist() for NumPy 2.x compatibility)
        self.buffer["states"].append(self.current_state.cpu().detach().tolist())
        self.buffer["actions"].append(chosen_theta)
        self.buffer["logprobs"].append(self.current_logprob)
        self.buffer["rewards"].append(utility)
        self.buffer["values"].append(val)
        self.buffer["dones"].append(float(done))

        # update when buffer full or episode done
        if len(self.buffer["states"]) >= self.buffer_size or done:
            self._ppo_update()
            for k in self.buffer:
                self.buffer[k] = []

    # ---------- ppo core ----------

    def _ppo_update(self):
        states_list = self.buffer["states"]
        # Pad states to max dimension if needed (for variable-length states from different info types)
        max_state_dim = max(len(s) for s in states_list) if states_list else self.state_dim
        states_padded = []
        for s in states_list:
            # Convert to list and ensure it's the right length
            s_list = list(s) if isinstance(s, (list, np.ndarray)) else [s]
            # Pad or truncate to state_dim
            if len(s_list) < self.state_dim:
                s_padded = s_list + [0.0] * (self.state_dim - len(s_list))
            else:
                s_padded = s_list[:self.state_dim]
            # Replace any NaN or inf with 0
            s_padded = [0.0 if not np.isfinite(x) else float(x) for x in s_padded]
            states_padded.append(s_padded)
        
        states = torch.tensor(np.array(states_padded), dtype=torch.float32)   # [T, state_dim]
        # Check for NaN in states
        if torch.isnan(states).any() or torch.isinf(states).any():
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        
        actions = torch.tensor(self.buffer["actions"], dtype=torch.float32).unsqueeze(1)  # [T,1]
        old_logprobs = torch.tensor(self.buffer["logprobs"], dtype=torch.float32)     # [T]
        rewards = torch.tensor(self.buffer["rewards"], dtype=torch.float32)           # [T]
        dones = torch.tensor(self.buffer["dones"], dtype=torch.float32)               # [T]

        # critic values at states (fixed for GAE computation)
        with torch.no_grad():
            values = self.critic(states).squeeze()                                    # [T]
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

        # standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Check and fix actor_log_std if it's NaN (can happen with exploding gradients)
        if torch.isnan(self.actor_log_std).any() or torch.isinf(self.actor_log_std).any():
            with torch.no_grad():
                self.actor_log_std.data.fill_(0.0)  # Reset to initial value

        for _ in range(self.update_epochs):
            # same distribution as at act-time (same transforms, same max_theta from states)
            dist, max_theta = self._policy_dist(states)
            
            # Clamp actions to valid range [0, max_theta] to prevent invalid log_prob computation
            # This handles cases where max_theta changed between sampling and update
            max_theta_vals = max_theta.squeeze(-1)  # [T]
            actions_squeezed = actions.squeeze(-1)  # [T]
            # Use torch.clamp with both min and max as numbers, or use torch.minimum/torch.maximum
            actions_clamped = torch.maximum(torch.zeros_like(actions_squeezed), 
                                           torch.minimum(actions_squeezed, max_theta_vals)).unsqueeze(-1)  # [T, 1]
            
            new_logprobs = dist.log_prob(actions_clamped).sum(dim=1)                           # [T]
            # entropy of base gaussian is fine for a signal (exact transformed entropy is messy)
            entropy = dist.base_dist.entropy().sum(dim=1).mean()

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
