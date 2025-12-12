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
        gamma: float = 0.99,
        total_auctions: int = 50,
        lr: float = 3e-4,
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

        # state = [value, budget_fraction, auctions_remaining_fraction]
        self.actor_mean = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # no sigmoid; we handle bounds via transforms
        )
        # global log std works; you can later make it state-dependent
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        self.critic = nn.Sequential(
            nn.Linear(3, 64),
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

    def get_state(self, value: float) -> torch.Tensor:
        return torch.tensor(
            [
                value,
                self.budget / self.initial_budget,
                self.auctions_remaining / self.total_auctions,
            ],
            dtype=torch.float32,
        )

    def draw_value(self) -> float:
        v = np.random.uniform(0.0, 1.0)
        self.current_value = v
        self.current_state = self.get_state(v)
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
        loc = self.actor_mean(states)                      # shape [..., 1]
        # clamp log_std to sane range to avoid pathological std
        log_std = self.actor_log_std.clamp(min=-5.0, max=2.0)
        std = log_std.exp().expand_as(loc)
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
        # first-price: you pay your own bid if you win
        price_paid = outcome.payments[self.agent_id]
        # price_paid = bid if won else 0.0

        # can't pay more than budget
        if won and price_paid > self.budget + 1e-12:
            too_expensive = True
            won = False
            price_paid = 0.0

        utility = outcome.utilities[self.agent_id] if too_expensive is False else 0.0
        # utility = self.compute_utility(value, won, price_paid)

        # if not np.isfinite(utility):
        #     print(f"[warning] non-finite utility for agent {self.agent_id}, skipping transition")
        #     return

        # budget & time
        
        self.budget -= price_paid
        self.auctions_remaining -= 1
        done = (self.budget <= 0.0) or (self.auctions_remaining == 0)

        # track (value, bid, utility, theta)
        self.history.append((value, bid, utility, chosen_theta))

        with torch.no_grad():
            val = self.critic(self.current_state.unsqueeze(0)).squeeze().item()

        # store transition
        self.buffer["states"].append(self.current_state.numpy())
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
        states = torch.tensor(np.array(self.buffer["states"]), dtype=torch.float32)   # [T,3]
        actions = torch.tensor(self.buffer["actions"], dtype=torch.float32).unsqueeze(1)  # [T,1]
        old_logprobs = torch.tensor(self.buffer["logprobs"], dtype=torch.float32)     # [T]
        rewards = torch.tensor(self.buffer["rewards"], dtype=torch.float32)           # [T]
        dones = torch.tensor(self.buffer["dones"], dtype=torch.float32)               # [T]

        # Check for NaN/inf in inputs
        if torch.isnan(states).any() or torch.isinf(states).any():
            print(f"Agent {self.agent_id}: NaN/inf in states, skipping update")
            return
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"Agent {self.agent_id}: NaN/inf in rewards, skipping update")
            return
        if torch.isnan(old_logprobs).any() or torch.isinf(old_logprobs).any():
            print(f"Agent {self.agent_id}: NaN/inf in old_logprobs, skipping update")
            return

        # Check network weights before update
        for name, param in self.actor_mean.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Agent {self.agent_id}: NaN/inf in {name} BEFORE update")
                return

        # critic values at states (fixed for GAE computation)
        with torch.no_grad():
            value_output = self.critic(states)  # [T, 1]
            # Squeeze only the last dimension to keep batch dimension (handle single sample case)
            values = value_output.squeeze(-1)  # [T] - always 1D even if T=1
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

        for _ in range(self.update_epochs):
            # same distribution as at act-time (same transforms, same max_theta from states)
            dist, _ = self._policy_dist(states)
            new_logprobs = dist.log_prob(actions).sum(dim=1)                           # [T]
            # entropy of base gaussian is fine for a signal (exact transformed entropy is messy)
            entropy = dist.base_dist.entropy().sum(dim=1).mean()

            new_values = self.critic(states).squeeze(-1)  # [T] - squeeze last dim only

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
