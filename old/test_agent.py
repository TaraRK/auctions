# the agent for MARL project
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim=1, action_dim=1):
        super(Actor, self).__init__()

        # Network to predict the mean (mu) of the action distribution
        self.mu_net = FFN(state_dim, action_dim)

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        # 1. Calculate the mean (mu)
        mu = self.mu_net(state)

        # 2. Exponentiate log_std to get the standard deviation (std)
        # Clamping is often used to prevent std from becoming too small (instability)
        std = torch.exp(self.log_std.clamp(-20, 2))

        # 3. Create the Normal distribution
        dist = Normal(mu, std)

        return dist

    def act(self, state):
        # The agent samples an action from the distribution
        dist = self.forward(state)
        action = dist.sample()

        # Calculate the log probability for the PPO objective
        log_prob = dist.log_prob(action).sum(axis=-1)

        action_clipped = torch.clamp(action, 0.0, 1.0)  # Clip to [0, 1]

        return action_clipped, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim=1):
        super(Critic, self).__init__()

        self.value_net = FFN(state_dim, 1)

    def forward(self, state):
        return self.value_net(state)


class PPOAgent:
    def __init__(
        self,
        actual_v,
        agent_id,
        state_dim=1,
        action_dim=1,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        gae_lambda=0.95,
    ):

        # Agent Identification and Auction Parameters
        self.id = agent_id
        self.actual_v = actual_v  # This will be set by the auction simulation logic
        self.v_i = None  # Current realized private value
        self.b_i = 0  # Last bid placed
        self.theta_i = 0  # Last shading factor (action) taken
        self.state_dim = state_dim

        # PPO Hyperparameters
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE parameter
        self.K_epochs = K_epochs  # Number of epochs to train on collected data
        self.eps_clip = eps_clip  # PPO clipping parameter

        # The Networks
        self.actor = Actor(
            state_dim, action_dim
        )  # Policy network: State -> Gaussian parameters
        self.critic = Critic(state_dim)  # Value network: State -> State Value V(s)

        # Optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Memory Buffer (Rollout)
        self.memory = []

    def get_state(self):
        """The state is the agent's current private value v_i."""
        # For this example, we assume observation noise is part of the environment,
        # but here we'll use the fundamental value as the state for simplicity.
        self.v_i = (
            self.actual_v + Normal(0, 0.01).sample().item()
        )  # Introduce small observation noise
        return np.array([self.v_i])

    def bid(self, state):
        """
        Uses the Actor network to determine the optimal shading factor (action).

        :param state: The observed state (v_i).
        :return: The bid b_i.
        """
        # Convert state to tensor for the network
        state_tensor = torch.FloatTensor(state.reshape(1, -1))

        # 1. Get Action (Shading Factor) and Log Probability
        with torch.no_grad():
            action_tensor, log_prob = self.actor.act(state_tensor)

        # 2. Store the action (shading factor)
        self.theta_i = action_tensor.item()

        # 3. Calculate the bid: b_i = v_i * (1 - theta_i)
        # Note: v_i is stored as the first (and only) element of the state array.
        v_i = state[0]
        self.b_i = v_i * (1 - self.theta_i)

        # The environment needs the bid and the log_prob for learning
        return self.b_i, self.theta_i, log_prob.item()

    def store_transition(self, state, action_theta, log_prob, reward, next_state, done):
        """
        Stores the collected experience from one round of the auction.
        """
        self.memory.append((state, action_theta, log_prob, reward, next_state, done))

    def learn(self):
        """
        Implements the full PPO update step on the collected batch of memory.
        This is the method from the previous answer, adapted here for context.
        """
        # --- PPO Learning Logic (GAE and Clipped Update) ---

        if not self.memory:
            return

        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)

        # 1. Convert to Tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # 2. Calculate Advantages and Returns (using the method from the previous response)
        # We need to define calculate_gae_and_returns as a helper function or method
        advantages, returns = self._calculate_gae_and_returns(
            states, rewards, dones, next_states
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO Optimization Loop
        for _ in range(self.K_epochs):
            # Actor Update (Clipped Loss)
            dist = self.actor(states)
            current_log_probs = dist.log_prob(actions).sum(axis=-1)
            ratio = torch.exp(current_log_probs - old_log_probs.detach())

            advantages_reshaped = advantages.unsqueeze(1)
            surr1 = ratio * advantages_reshaped
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages_reshaped
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Update (Value Loss)
            current_values = self.critic(states)
            critic_loss = nn.MSELoss()(current_values, returns.unsqueeze(1))

            # Backpropagation and Optimization
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.optimizer_critic.step()

        # 4. Clear memory
        self.memory = []

        return actor_loss.item(), critic_loss.item()  # Return losses for logging

    def _calculate_gae_and_returns(self, states, rewards, dones, next_states):
        """Helper function for calculating GAE, same as previous response."""
        values = self.critic(states).squeeze().detach().numpy()
        next_values = self.critic(next_states).squeeze().detach().numpy()

        advantages = np.zeros_like(rewards.numpy())
        last_gae = 0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t].item()
                + self.gamma * next_values[t].item() * (1 - dones[t].item())
                - values[t].item()
            )
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * (
                1 - dones[t].item()
            )
            advantages[t] = last_gae

        returns = advantages + values
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
