import numpy as np
import torch
import torch.nn as nn
from agent import Agent
from torch import optim

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
        
        # state is now [value, budget_fraction, auctions_remaining_fraction]
        self.actor_mean = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs fraction of value to bid
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        
        self.critic = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.actor_mean.parameters()) + 
            list(self.critic.parameters()) + 
            [self.actor_log_std],
            lr=lr
        )
        
        self.buffer = {
            'values': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': [],
            'states': []
        }
        
        self.current_value = None
        self.current_state = None
        self.current_theta = None
        self.current_logprob = None
        
    def reset(self):
        """reset budget and auction count for new episode"""
        self.budget = self.initial_budget
        self.auctions_remaining = self.total_auctions
        
    def get_state(self, value: float) -> torch.Tensor:
        """state = [value, budget/initial_budget, auctions_remaining/total]"""
        return torch.tensor([
            value,
            self.budget / self.initial_budget,
            self.auctions_remaining / self.total_auctions
        ], dtype=torch.float32)
    
    def draw_value(self) -> float:
        v = np.random.uniform(0.0, 1.0)
        self.current_value = v
        self.current_state = self.get_state(v)
        return v
    
    def choose_theta(self) -> float:

        # compute feasible theta range
        max_feasible_bid = min(self.current_value, self.budget)
        max_feasible_theta = max_feasible_bid / (self.current_value + 1e-8)
        max_feasible_theta = min(max_feasible_theta, 1.0)
        
        with torch.no_grad():
            raw_mean = self.actor_mean(self.current_state)  # in [0,1]
            
            # squash mean into feasible range [0, max_feasible_theta]
            mean = raw_mean * max_feasible_theta
            
            std = torch.exp(self.actor_log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0, max_feasible_theta)
            logprob = dist.log_prob(action).sum()
        
        self.current_theta = action.item()
        self.current_logprob = logprob.item()
        self.theta = self.current_theta
        return self.current_theta
    
        # with torch.no_grad():
        #     mean = self.actor_mean(self.current_state)  # theta in [0,1]
        #     std = torch.exp(self.actor_log_std)
        #     dist = torch.distributions.Normal(mean, std)
        #     sampled_action = dist.sample()
        #     sampled_action = torch.clamp(sampled_action, 0, 1)
        #     logprob = dist.log_prob(sampled_action).sum()
            
        #     # compute actual bid respecting budget
        #     bid = sampled_action.item() * self.current_value
        #     bid = min(bid, self.budget)
        #     actual_theta = bid / (self.current_value + 1e-8)
        
        # # store sampled action for policy gradient
        # self.current_theta = actual_theta  # what we actually bid
        # self.sampled_theta = sampled_action.item()  # what network output
        # self.current_logprob = logprob.item()
        # self.theta = actual_theta
        # return actual_theta

    
    def update(self, value: float, chosen_theta: float, outcome):
        bid = value * chosen_theta
        won = (outcome.winner_idx == self.agent_id)
        price_paid = outcome.winning_bid if won else 0.0
        
        # can't pay more than budget
        if won and price_paid > self.budget:
            won = False  # forfeit if can't afford
            price_paid = 0.0
        
        utility = self.compute_utility(value, won, price_paid)
        
        # update budget and auctions
        if won:
            self.budget -= price_paid
        self.auctions_remaining -= 1
        
        done = (self.budget <= 0 or self.auctions_remaining == 0)
        
        self.history.append((value, bid, utility, chosen_theta))
        
        # store transition
        with torch.no_grad():
            val = self.critic(self.current_state).item()
        
        self.buffer['states'].append(self.current_state.numpy())
        self.buffer['actions'].append(chosen_theta)
        # self.buffer['actions'].append(self.sampled_theta)
        self.buffer['logprobs'].append(self.current_logprob)
        self.buffer['rewards'].append(utility)
        self.buffer['values'].append(val)
        self.buffer['dones'].append(done)
        
        # update when buffer full or episode done
        if len(self.buffer['states']) >= self.buffer_size or done:
            self._ppo_update()
            for k in self.buffer:
                self.buffer[k] = []
            
            # if done:
            #     self.reset()
    
    def _ppo_update(self):
        states = torch.tensor(np.array(self.buffer['states']), dtype=torch.float32)
        actions = torch.tensor(self.buffer['actions'], dtype=torch.float32).unsqueeze(1)
        old_logprobs = torch.tensor(self.buffer['logprobs'], dtype=torch.float32)
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32)
        old_values = torch.tensor(self.buffer['values'], dtype=torch.float32)
        dones = torch.tensor(self.buffer['dones'], dtype=torch.float32)
        
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = 0 if dones[t] else values[t]
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.update_epochs):
            mean = self.actor_mean(states)
            std = torch.exp(self.actor_log_std)
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            
            new_values = self.critic(states).squeeze()
            
            ratio = torch.exp(new_logprobs - old_logprobs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()
            entropy_loss = entropy.mean()
            
            loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_mean.parameters()) + 
                list(self.critic.parameters()) + 
                [self.actor_log_std],
                0.5
            )
            self.optimizer.step()