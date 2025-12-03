# from typing import List
# import numpy as np
# from agent import Agent

# class AuctionOutcome:
#     winner_idx: int
#     winning_bid: float
#     all_bids: np.ndarray
#     utilities: np.ndarray

# class FirstPriceAuction:
#     def __init__(self, n_agents: int):
#         self.n_agents = n_agents
    
#     def run_auction(self, bids: np.ndarray) -> AuctionOutcome:
#         """run single auction round, return outcome"""
#         winner_idx = np.argmax(bids)
#         winning_bid = bids[winner_idx]
        
#         utilities = np.zeros(self.n_agents)
#         # winner gets value - payment, losers get 0 (already initialized)
        
#         return AuctionOutcome(
#             winner_idx=winner_idx,
#             winning_bid=winning_bid,
#             all_bids=bids.copy(),
#             utilities=utilities  # agents compute their own utility since they know their v
#         )
        

# class QLearningAgent(Agent):
#     def __init__(
#         self,
#         agent_id: int,
#         n_value_bins: int = 10,
#         theta_options: np.ndarray = None,
#         alpha: float = 0.1,    
#         gamma: float = 0.0,     
#         epsilon: float = 0.1    
#     ):
#         super().__init__(agent_id)
#         self.n_value_bins = n_value_bins
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.theta = 0

#         if theta_options is None:
#             theta_options = np.linspace(0, 1.0, 500)
#         self.theta_options = theta_options

#         self.Q = np.zeros((self.n_value_bins, len(self.theta_options)))

#         self.current_value = None
#         self.current_state_idx = None
#         self.current_action_idx = None

#         self.history = []  

#     def _value_to_state(self, v: float) -> int:
#         idx = int(v * self.n_value_bins)
#         if idx == self.n_value_bins: 
#             idx = self.n_value_bins - 1
#         return idx

#     def draw_value(self) -> float:
#         v = np.random.uniform(0.0, 1.0)
#         self.current_value = v
#         self.current_state_idx = self._value_to_state(v)
#         return v

#     def choose_theta(self) -> float:
#         assert self.current_state_idx is not None, "call draw_value() before choose_theta()"
#         s = self.current_state_idx

#         if np.random.rand() < self.epsilon:
#             a = np.random.randint(len(self.theta_options))
#         else:
#             a = np.argmax(self.Q[s])

#         self.current_action_idx = a
#         self.theta = self.theta_options[a]
#         return self.theta

#     def update(self, value: float, chosen_theta: float, outcome):
#         # compute realized utility
#         bid = value * chosen_theta
#         won = (outcome.winner_idx == self.agent_id)
#         utility = self.compute_utility(value, won, outcome.winning_bid)

#         # log for diagnostics
#         self.history.append((value, bid, utility, chosen_theta))

#         # Q-learning update
#         s = self._value_to_state(value)
#         a = self.current_action_idx
#         r = utility

#         # single-step auction, treat next-state value as irrelevant (gamma=0)
#         # target = r + gamma * max_a' Q(s', a')  -> here gamma=0 => target = r
#         target = r 
#         self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * target
#         # if you later want gamma>0, you could sample/approximate s_next here

#         # optional: you could adapt epsilon over time if you want annealing
#         # self.epsilon = max(self.epsilon * 0.9999, 0.01)

# if __name__ == "__main__":
#     def run_simulation_with_logging(n_agents: int, n_rounds: int):
#         auction = FirstPriceAuction(n_agents)
#         agents: List[Agent] = []
#         for i in range(n_agents):
#             # agents.append(Agent(i, learning_rate=0.01))
#             # agents.append(RegretMatchingAgent(i))
#             agents.append(QLearningAgent(i, n_value_bins=10, alpha=0.1, gamma=0.0, epsilon=0.1))
#             # agents.append(PPOAgent(i, budget=10.0))
        
#         # tracking
#         theta_hist = [[] for _ in range(n_agents)]
#         avg_theta_hist = []
#         efficiency_hist = []  # did highest-value agent win?
#         revenue_hist = []
#         for round_idx in range(n_rounds):
#             # each agent draws value and computes bid
#             values = np.array([agent.draw_value() for agent in agents])
#             thetas = [agent.choose_theta() for agent in agents]  # new: sample theta
#             bids = np.array([values[i] * thetas[i] for i in range(n_agents)])
#             # bids = np.array([agents[i].compute_bid(values[i]) for i in range(n_agents)])
            
#             # run auction
#             outcome = auction.run_auction(bids)
            
#             # agents update based on outcome
#             for i, agent in enumerate(agents):
#                 agent.update(values[i], thetas[i], outcome)
            
#             # track metrics
#             for i in range(n_agents):
#                 theta_hist[i].append(thetas[i])
            
#             avg_theta_hist.append(np.mean(thetas))
            
#             # efficiency: did highest-value agent win?
#             highest_value_idx = np.argmax(values)
#             efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)
            
#             # revenue: winning bid
#             revenue_hist.append(outcome.winning_bid)
            
#             # log stuff every k rounds
#             if round_idx % 1000 == 0:
#                 avg_theta = np.mean([a.theta for a in agents])
#                 var_theta = np.var([a.theta for a in agents])
#                 print(f"round {round_idx}: avg_theta={avg_theta:.3f}, var_theta={var_theta:.4f}, theory={(n_agents - 1)/n_agents:.3f}")
#                 # i think the theory is 1/n, but it could be different!

from typing import List
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------

class Agent:
    """Minimal base Agent to make QLearningAgent self-contained.
    If you already have Agent defined elsewhere, you can drop this."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.theta = 0.0

    def compute_utility(self, value: float, won: bool, winning_bid: float) -> float:
        # First-price private-value: utility = v - p if win, 0 otherwise
        return value - winning_bid if won else 0.0


@dataclass
class AuctionOutcome:
    winner_idx: int
    winning_bid: float
    all_bids: np.ndarray
    utilities: np.ndarray


class FirstPriceAuction:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
    
    def run_auction(self, bids: np.ndarray) -> AuctionOutcome:
        """run single auction round, return outcome"""
        winner_idx = int(np.argmax(bids))
        winning_bid = float(bids[winner_idx])
        
        utilities = np.zeros(self.n_agents)
        # winner gets value - payment, losers get 0 (agents compute this themselves)

        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            utilities=utilities
        )


# ---------------------------------------------------------------------
# Q-learning agent with budget: state = (value_bin, budget_bin)
# ---------------------------------------------------------------------

class QLearningAgent(Agent):
    def __init__(
        self,
        agent_id: int,
        n_value_bins: int = 1000,
        n_budget_bins: int = 1000,
        theta_options: np.ndarray = None,
        alpha: float = 0.1,     # learning rate
        gamma: float = 0.9,     # discount factor > 0 now
        epsilon: float = 0.1,   # epsilon-greedy
        initial_budget: float = 10.0
    ):
        super().__init__(agent_id)
        self.n_value_bins = n_value_bins
        self.n_budget_bins = n_budget_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if theta_options is None:
            theta_options = np.linspace(0.0, 1.0, 1000)
        self.theta_options = theta_options

        # Q[state_v, state_b, action]
        self.Q = np.zeros((self.n_value_bins, self.n_budget_bins, len(self.theta_options)))

        # budget
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget

        # current step
        self.current_value = None
        self.current_v_idx = None
        self.current_b_idx = None
        self.current_action_idx = None

        # last transition (for Q update once next state is known)
        self.last_state = None        # (v_idx, b_idx)
        self.last_action = None       # int
        self.last_reward = 0.0        # float

        self.history = []  # (v, bid, utility, theta, remaining_budget)

    # -------- discretization helpers --------

    def _value_to_state(self, v: float) -> int:
        v = max(0.0, min(1.0, v))
        idx = int(v * self.n_value_bins)
        if idx == self.n_value_bins:
            idx = self.n_value_bins - 1
        return idx

    def _budget_to_state(self, budget: float) -> int:
        if self.initial_budget <= 0.0:
            return 0
        frac = max(0.0, min(1.0, budget / self.initial_budget))
        idx = int(frac * self.n_budget_bins)
        if idx == self.n_budget_bins:
            idx = self.n_budget_bins - 1
        return idx

    # -------- episode / round management --------

    def reset_episode(self, budget: float = None):
        """Call once at the start of a simulation/episode."""
        if budget is None:
            budget = self.initial_budget
        self.initial_budget = budget
        self.remaining_budget = budget

        self.current_value = None
        self.current_v_idx = None
        self.current_b_idx = None
        self.current_action_idx = None

        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0

        self.history = []

    def _update_q_from_last_transition(self, next_v_idx: int, next_b_idx: int):
        """Q-learning update: uses last_state, last_action, last_reward and next_state."""
        if self.last_state is None or self.last_action is None:
            return  # nothing to update yet (first round)

        v_prev, b_prev = self.last_state
        a_prev = self.last_action
        r_prev = self.last_reward

        # bootstrap on next state
        best_next = np.max(self.Q[next_v_idx, next_b_idx])
        target = r_prev + self.gamma * best_next
        q_old = self.Q[v_prev, b_prev, a_prev]
        self.Q[v_prev, b_prev, a_prev] = (1 - self.alpha) * q_old + self.alpha * target

    def begin_round(self) -> float:
        """
        Called at the start of each auction round.
        - Samples a new value v_t
        - Defines current state s_t = (v_bin, budget_bin)
        - Uses this s_t as the 'next_state' to update Q for (s_{t-1}, a_{t-1}, r_{t-1})
        """
        # sample value only if we still have budget; otherwise effectively drop out
        if self.remaining_budget > 0.0:
            v = np.random.uniform(0.0, 1.0)
        else:
            v = 0.0  # no budget, will bid zero

        v_idx = self._value_to_state(v)
        b_idx = self._budget_to_state(self.remaining_budget)

        # use (v_idx, b_idx) as s_{t+1} to update previous transition
        self._update_q_from_last_transition(v_idx, b_idx)

        # set current state
        self.current_value = v
        self.current_v_idx = v_idx
        self.current_b_idx = b_idx
        self.current_action_idx = None

        return v

    # -------- action selection --------

    def choose_theta(self) -> float:
        assert self.current_v_idx is not None and self.current_b_idx is not None, \
            "call begin_round() before choose_theta()"

        # out of money => effectively don't bid
        if self.remaining_budget <= 0.0:
            self.theta = 0.0
            self.current_action_idx = None
            return self.theta

        v_idx, b_idx = self.current_v_idx, self.current_b_idx

        if np.random.rand() < self.epsilon:
            a = np.random.randint(len(self.theta_options))
        else:
            a = int(np.argmax(self.Q[v_idx, b_idx]))

        self.current_action_idx = a
        self.theta = float(self.theta_options[a])

        # store state-action for upcoming transition
        self.last_state = (v_idx, b_idx)
        self.last_action = a

        return self.theta

    # -------- outcome + reward --------

    def observe_outcome(self, value: float, chosen_theta: float, outcome: AuctionOutcome):
        """
        Called once per round after the auction outcome is known.
        Stores reward for later Q-update (done at next begin_round or final_update).
        Also updates remaining budget.
        """
        bid = value * chosen_theta
        won = (outcome.winner_idx == self.agent_id)
        utility = self.compute_utility(value, won, outcome.winning_bid)

        if won:
            payment = outcome.winning_bid

            # if you want to enforce hard budget constraints, you can impose a penalty
            # for overspending; here we clamp payment and penalize any overshoot.
            if payment > self.remaining_budget + 1e-8:
                # heavy negative penalty for trying to overspend
                utility -= (payment - self.remaining_budget) * 10.0
                payment = self.remaining_budget

            self.remaining_budget = max(0.0, self.remaining_budget - payment)

        self.history.append((value, bid, utility, chosen_theta, self.remaining_budget))
        self.last_reward = utility

    def final_update(self):
        """Call once at the very end of an episode to update using a terminal state (no bootstrap)."""
        if self.last_state is None or self.last_action is None:
            return

        v_prev, b_prev = self.last_state
        a_prev = self.last_action
        r_prev = self.last_reward

        target = r_prev  # terminal, no future value
        q_old = self.Q[v_prev, b_prev, a_prev]
        self.Q[v_prev, b_prev, a_prev] = (1 - self.alpha) * q_old + self.alpha * target

        self.last_state = None
        self.last_action = None


# ---------------------------------------------------------------------
# Simulation loop using the new budget-aware QLearningAgent
# ---------------------------------------------------------------------

def run_simulation_with_logging(
    n_agents: int,
    n_rounds: int,
    initial_budget: float = 10.0
):
    auction = FirstPriceAuction(n_agents)
    agents: List[QLearningAgent] = []

    for i in range(n_agents):
        agent = QLearningAgent(
            i,
            n_value_bins=10,
            n_budget_bins=10,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            initial_budget=initial_budget
        )
        agent.reset_episode(initial_budget)
        agents.append(agent)

    theta_hist = [[] for _ in range(n_agents)]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    budget_hist = [[] for _ in range(n_agents)]

    for round_idx in range(n_rounds):
        # each agent begins round (this also performs the Q update for the previous transition)
        values = np.array([agent.begin_round() for agent in agents])
        thetas = [agent.choose_theta() for agent in agents]

        bids = np.array([values[i] * thetas[i] for i in range(n_agents)])
        outcome = auction.run_auction(bids)

        # agents observe outcome (reward + budget transition)
        for i, agent in enumerate(agents):
            agent.observe_outcome(values[i], thetas[i], outcome)

        # tracking
        for i in range(n_agents):
            theta_hist[i].append(thetas[i])
            budget_hist[i].append(agents[i].remaining_budget)

        avg_theta_hist.append(np.mean(thetas))

        # efficiency: did highest-value agent win?
        highest_value_idx = int(np.argmax(values))
        efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)

        # revenue: winning bid
        revenue_hist.append(outcome.winning_bid)

        if round_idx % 1000 == 0:
            avg_theta = np.mean([a.theta for a in agents])
            var_theta = np.var([a.theta for a in agents])
            print(
                f"round {round_idx}: "
                f"avg_theta={avg_theta:.3f}, var_theta={var_theta:.4f}, "
                f"theory={(n_agents - 1)/n_agents:.3f}"
            )

    # terminal Q updates (no bootstrap)
    for agent in agents:
        agent.final_update()

    return {
        "theta_hist": theta_hist,
        "avg_theta_hist": avg_theta_hist,
        "efficiency_hist": efficiency_hist,
        "revenue_hist": revenue_hist,
        "budget_hist": budget_hist,
        "agents": agents,
    }


if __name__ == "__main__":
    _ = run_simulation_with_logging(n_agents=10, n_rounds=50000, initial_budget=50000.0)
