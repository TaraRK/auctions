import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
from matplotlib import pyplot as plt


class AuctionType(Enum):
    FIRST_PRICE = "first_price"
    ASCENDING_CLOCK = "ascending_clock"
    DESCENDING_CLOCK = "descending_clock"


class InformationType(Enum):
    MINIMAL = "minimal"  # Only win/loss
    WINNER = "winner"  # Winner learns losing bid(s)
    LOSER = "loser"  # Loser learns winning bid
    FULL_TRANSPARENCY = "full_transparency"  # Everyone sees all bids
    FULL_REVELATION = "full_revelation"  # Sealed-bid: All info revealed after auction (all bids, all values, all payments)


@dataclass
class AgentInformation:
    won: bool
    own_bid: float  # For first-price: bid amount; For clock: dropout price
    own_value: float
    winning_bid: Optional[float] = None  # Only if revealed
    losing_bids: Optional[np.ndarray] = None  # Only if revealed (for winner)
    all_bids: Optional[np.ndarray] = None  # Only if full transparency
    # Clock auction specific
    dropout_round: Optional[int] = None  # Round when agent dropped out (clock auctions)
    dropout_order: Optional[List[int]] = None  # Order of dropouts (if revealed)
    final_price: Optional[float] = None  # Final clearing price (clock auctions)


@dataclass
class AuctionOutcome:
    winner_idx: int
    winning_bid: float  # For clock auctions: final clearing price
    all_bids: np.ndarray  # For first-price: all bids; For clock: dropout prices
    utilities: np.ndarray
    agent_info: Dict[int, AgentInformation]  # Information revealed to each agent
    auction_type: AuctionType = AuctionType.FIRST_PRICE


class FirstPriceAuction:
    def __init__(self, n_agents: int, info_type: InformationType = InformationType.MINIMAL):
        self.n_agents = n_agents
        self.info_type = info_type
    
    def _reveal_information(self, bids: np.ndarray, values: np.ndarray, winner_idx: int) -> Dict[int, AgentInformation]:
        agent_info = {}
        winning_bid = bids[winner_idx]
        
        for i in range(self.n_agents):
            won = (i == winner_idx)
            own_bid = bids[i]
            own_value = values[i]
            
            if self.info_type == InformationType.MINIMAL:
                # Only win/loss
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_bid,
                    own_value=own_value
                )
            
            elif self.info_type == InformationType.WINNER:
                if won:
                    # Winner learns all losing bids
                    losing_bids = bids.copy()
                    losing_bids[i] = -1  # mask own bid
                    losing_bids = losing_bids[losing_bids >= 0]  # remove masked
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value,
                        winning_bid=winning_bid,
                        losing_bids=losing_bids
                    )
                else:
                    # Loser learns nothing
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value
                    )
            
            elif self.info_type == InformationType.LOSER:
                if won:
                    # Winner learns nothing
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value
                    )
                else:
                    # Loser learns winning bid
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_bid,
                        own_value=own_value,
                        winning_bid=winning_bid
                    )
            
            elif self.info_type == InformationType.FULL_TRANSPARENCY:
                # Everyone sees all bids
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_bid,
                    own_value=own_value,
                    winning_bid=winning_bid,
                    all_bids=bids.copy()
                )
            
            elif self.info_type == InformationType.FULL_REVELATION:
                # Sealed-bid: All information revealed after auction
                # Everyone sees all bids (same as full transparency)
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_bid,
                    own_value=own_value,
                    winning_bid=winning_bid,
                    all_bids=bids.copy()
                )
        
        return agent_info
    
    def run_auction(self, bids: np.ndarray, values: np.ndarray) -> AuctionOutcome:
        """run single auction round, return outcome with information revelation"""
        winner_idx = np.argmax(bids)
        winning_bid = bids[winner_idx]
        
        utilities = np.zeros(self.n_agents)
        # winner gets value - payment, losers get 0 (already initialized)
        
        agent_info = self._reveal_information(bids, values, winner_idx)
        
        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            utilities=utilities,  # agents compute their own utility since they know their v
            agent_info=agent_info,
            auction_type=AuctionType.FIRST_PRICE
        )


class AscendingClockAuction:
    def __init__(
        self,
        n_agents: int,
        info_type: InformationType = InformationType.MINIMAL,
        price_step: float = 0.01,
        max_price: float = 1.0
    ):
        self.n_agents = n_agents
        self.info_type = info_type
        self.price_step = price_step
        self.max_price = max_price
    
    def _reveal_information(
        self,
        dropout_prices: np.ndarray,
        values: np.ndarray,
        winner_idx: int,
        dropout_order: List[int],
        final_price: float
    ) -> Dict[int, AgentInformation]:
        """Reveal information to agents based on info_type"""
        agent_info = {}
        
        for i in range(self.n_agents):
            won = (i == winner_idx)
            own_dropout_price = dropout_prices[i]
            own_value = values[i]
            
            if self.info_type == InformationType.MINIMAL:
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_dropout_price,
                    own_value=own_value,
                    dropout_round=dropout_order.index(i) if i in dropout_order else None,
                    final_price=final_price if won else None
                )
            
            elif self.info_type == InformationType.WINNER:
                if won:
                    # Winner learns all dropout prices
                    losing_dropouts = dropout_prices.copy()
                    losing_dropouts[i] = -1
                    losing_dropouts = losing_dropouts[losing_dropouts >= 0]
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        winning_bid=final_price,
                        losing_bids=losing_dropouts,
                        dropout_round=dropout_order.index(i),
                        dropout_order=dropout_order,
                        final_price=final_price
                    )
                else:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        dropout_round=dropout_order.index(i) if i in dropout_order else None
                    )
            
            elif self.info_type == InformationType.LOSER:
                if won:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        dropout_round=dropout_order.index(i),
                        final_price=final_price
                    )
                else:
                    # Loser learns final price
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        winning_bid=final_price,
                        dropout_round=dropout_order.index(i) if i in dropout_order else None,
                        final_price=final_price
                    )
            
            elif self.info_type == InformationType.FULL_TRANSPARENCY:
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_dropout_price,
                    own_value=own_value,
                    winning_bid=final_price,
                    all_bids=dropout_prices.copy(),
                    dropout_round=dropout_order.index(i) if i in dropout_order else None,
                    dropout_order=dropout_order,
                    final_price=final_price
                )
        
        return agent_info
    
    def run_auction(
        self,
        dropout_decisions: callable,  # Function that takes (price, active_agents) -> List[bool]
        values: np.ndarray,
        max_willingness_to_pay: np.ndarray  # Maximum price each agent is willing to pay
    ) -> AuctionOutcome:
        """
        Run ascending clock auction.
        
        Args:
            dropout_decisions: Function that returns dropout decisions for each agent
            values: Private values of agents
            max_willingness_to_pay: Maximum price each agent is willing to pay (v * (1 - theta))
        """
        active_agents = set(range(self.n_agents))
        current_price = 0.0
        dropout_prices = np.full(self.n_agents, -1.0)  # -1 means didn't drop out yet
        dropout_order = []
        
        # Run clock rounds
        while len(active_agents) > 1 and current_price < self.max_price:
            # Get dropout decisions from agents
            should_dropout = dropout_decisions(current_price, list(active_agents), values, max_willingness_to_pay)
            
            # Process dropouts
            new_dropouts = []
            for agent_idx in list(active_agents):
                if should_dropout[agent_idx] or current_price >= max_willingness_to_pay[agent_idx]:
                    active_agents.remove(agent_idx)
                    dropout_prices[agent_idx] = current_price
                    dropout_order.append(agent_idx)
                    new_dropouts.append(agent_idx)
            
            # If only one agent left, they win
            if len(active_agents) == 1:
                winner_idx = list(active_agents)[0]
                final_price = current_price
                dropout_prices[winner_idx] = current_price  # Winner's dropout price = final price
                break
            
            # Increase price
            current_price += self.price_step
        
        # If we hit max price with multiple agents, winner is random among active
        if len(active_agents) > 1:
            winner_idx = np.random.choice(list(active_agents))
            final_price = current_price
            dropout_prices[winner_idx] = final_price
        elif len(active_agents) == 1:
            winner_idx = list(active_agents)[0]
            final_price = current_price
        
        utilities = np.zeros(self.n_agents)
        utilities[winner_idx] = values[winner_idx] - final_price
        
        # Reveal information
        agent_info = self._reveal_information(
            dropout_prices, values, winner_idx, dropout_order, final_price
        )
        
        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=final_price,
            all_bids=dropout_prices,
            utilities=utilities,
            agent_info=agent_info,
            auction_type=AuctionType.ASCENDING_CLOCK
        )


class DescendingClockAuction:
    """
    Descending clock auction: price starts high and decreases.
    Agents decide when to drop out (when price falls below their willingness to pay).
    First agent to drop out wins at that price.
    """
    def __init__(
        self,
        n_agents: int,
        info_type: InformationType = InformationType.MINIMAL,
        price_step: float = 0.01,
        min_price: float = 0.0
    ):
        self.n_agents = n_agents
        self.info_type = info_type
        self.price_step = price_step
        self.min_price = min_price
    
    def _reveal_information(
        self,
        dropout_prices: np.ndarray,
        values: np.ndarray,
        winner_idx: int,
        dropout_order: List[int],
        final_price: float
    ) -> Dict[int, AgentInformation]:
        """Reveal information to agents based on info_type"""
        agent_info = {}
        
        for i in range(self.n_agents):
            won = (i == winner_idx)
            own_dropout_price = dropout_prices[i]
            own_value = values[i]
            
            if self.info_type == InformationType.MINIMAL:
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_dropout_price,
                    own_value=own_value,
                    dropout_round=dropout_order.index(i) if i in dropout_order else None,
                    final_price=final_price if won else None
                )
            
            elif self.info_type == InformationType.WINNER:
                if won:
                    # Winner learns all dropout prices
                    losing_dropouts = dropout_prices.copy()
                    losing_dropouts[i] = -1
                    losing_dropouts = losing_dropouts[losing_dropouts >= 0]
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        winning_bid=final_price,
                        losing_bids=losing_dropouts,
                        dropout_round=dropout_order.index(i),
                        dropout_order=dropout_order,
                        final_price=final_price
                    )
                else:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        dropout_round=dropout_order.index(i) if i in dropout_order else None
                    )
            
            elif self.info_type == InformationType.LOSER:
                if won:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        dropout_round=dropout_order.index(i),
                        final_price=final_price
                    )
                else:
                    agent_info[i] = AgentInformation(
                        won=won,
                        own_bid=own_dropout_price,
                        own_value=own_value,
                        winning_bid=final_price,
                        dropout_round=dropout_order.index(i) if i in dropout_order else None,
                        final_price=final_price
                    )
            
            elif self.info_type == InformationType.FULL_TRANSPARENCY:
                agent_info[i] = AgentInformation(
                    won=won,
                    own_bid=own_dropout_price,
                    own_value=own_value,
                    winning_bid=final_price,
                    all_bids=dropout_prices.copy(),
                    dropout_round=dropout_order.index(i) if i in dropout_order else None,
                    dropout_order=dropout_order,
                    final_price=final_price
                )
        
        return agent_info
    
    def run_auction(
        self,
        dropout_decisions: callable,
        values: np.ndarray,
        min_willingness_to_pay: np.ndarray  # Minimum price each agent is willing to pay
    ) -> AuctionOutcome:
        """
        Run descending clock auction.
        
        First agent to drop out wins at that price.
        """
        active_agents = set(range(self.n_agents))
        current_price = 1.0  # Start high
        dropout_prices = np.full(self.n_agents, -1.0)
        dropout_order = []
        
        # Run clock rounds
        while len(active_agents) > 0 and current_price >= self.min_price:
            # Get dropout decisions
            should_dropout = dropout_decisions(current_price, list(active_agents), values, min_willingness_to_pay)
            
            # Process dropouts
            new_dropouts = []
            for agent_idx in list(active_agents):
                if should_dropout[agent_idx] or current_price <= min_willingness_to_pay[agent_idx]:
                    active_agents.remove(agent_idx)
                    dropout_prices[agent_idx] = current_price
                    dropout_order.append(agent_idx)
                    new_dropouts.append(agent_idx)
                    
                    # First dropout wins in descending clock
                    if len(new_dropouts) == 1:
                        winner_idx = agent_idx
                        final_price = current_price
                        break
            
            if len(new_dropouts) > 0:
                break
            
            # Decrease price
            current_price -= self.price_step
        
        # If no one dropped out, random winner at min price
        if len(dropout_order) == 0:
            winner_idx = np.random.choice(list(range(self.n_agents)))
            final_price = self.min_price
            dropout_prices[winner_idx] = final_price
        else:
            winner_idx = dropout_order[0]
            final_price = dropout_prices[winner_idx]
        
        utilities = np.zeros(self.n_agents)
        utilities[winner_idx] = values[winner_idx] - final_price
        
        # Reveal information
        agent_info = self._reveal_information(
            dropout_prices, values, winner_idx, dropout_order, final_price
        )
        
        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=final_price,
            all_bids=dropout_prices,
            utilities=utilities,
            agent_info=agent_info,
            auction_type=AuctionType.DESCENDING_CLOCK
        )


class BanditAgent:
    """
    Multi-armed bandit agent using UCB1 algorithm for auction bidding.
    
    The agent discretizes the action space (theta values) and treats each
    as an arm in a multi-armed bandit problem.
    """
    def __init__(
        self,
        agent_id: int,
        theta_options: np.ndarray = None,
        n_value_bins: int = 10,
        c: float = 2.0,  # UCB exploration constant
        initial_pulls: int = 1  # Number of times to pull each arm initially
    ):
        self.agent_id = agent_id
        
        # Discretize theta (shading factor) space
        if theta_options is None:
            theta_options = np.linspace(0.0, 1.0, 1000)  # Match QLearningAgent (1000 options)
        self.theta_options = theta_options
        self.n_arms = len(theta_options)
        
        # Discretize value space for state-dependent learning
        self.n_value_bins = n_value_bins
        
        # UCB1 statistics: Q-values, counts per (state, arm)
        self.Q = np.zeros((n_value_bins, self.n_arms))
        self.counts = np.ones((n_value_bins, self.n_arms)) * initial_pulls  # Start with 1 pull each
        self.total_pulls = np.ones((n_value_bins, self.n_arms)) * initial_pulls
        
        # UCB parameters
        self.c = c
        
        # Current state tracking
        self.current_value = None
        self.current_state_idx = None
        self.current_arm_idx = None
        
        # History for analysis
        self.history = []  # (value, bid, utility, theta, info_received)
    
    def _value_to_state(self, v: float) -> int:
        """Convert continuous value to discrete state bin"""
        idx = int(v * self.n_value_bins)
        if idx >= self.n_value_bins:
            idx = self.n_value_bins - 1
        return idx
    
    def draw_value(self) -> float:
        """Draw private value for this auction"""
        v = np.random.uniform(0.0, 1.0)
        self.current_value = v
        self.current_state_idx = self._value_to_state(v)
        return v
    
    def choose_theta(self) -> float:
        """Choose theta using UCB1 algorithm"""
        assert self.current_state_idx is not None, "call draw_value() before choose_theta()"
        s = self.current_state_idx
        
        # UCB1: choose arm with highest upper confidence bound
        ucb_values = self.Q[s] + self.c * np.sqrt(
            np.log(self.total_pulls[s].sum() + 1) / (self.counts[s] + 1e-10)
        )
        
        # Break ties randomly
        best_arms = np.where(ucb_values == ucb_values.max())[0]
        arm_idx = np.random.choice(best_arms)
        
        self.current_arm_idx = arm_idx
        theta = self.theta_options[arm_idx]
        return theta
    
    def compute_utility(self, value: float, won: bool, price_paid: float) -> float:
        """Compute utility: v - price if won, else 0"""
        return (value - price_paid) if won else 0.0
    
    def should_dropout_ascending(self, current_price: float, value: float, max_willingness: float) -> bool:
        """
        Decide whether to drop out in ascending clock auction.
        Drop out when price exceeds willingness to pay.
        """
        return current_price >= max_willingness
    
    def should_dropout_descending(self, current_price: float, value: float, min_willingness: float) -> bool:
        """
        Decide whether to drop out in descending clock auction.
        Drop out when price falls below willingness to pay.
        """
        return current_price <= min_willingness
    
    def update(self, value: float, chosen_theta: float, outcome: AuctionOutcome):
        """
        Update bandit statistics based on auction outcome and revealed information.
        
        The agent only uses the information revealed to them (in outcome.agent_info).
        """
        info = outcome.agent_info[self.agent_id]
        bid = value * chosen_theta
        won = info.won
        utility = self.compute_utility(value, won, outcome.winning_bid)
        
        # Log history
        self.history.append((
            value, bid, utility, chosen_theta,
            info.winning_bid, info.losing_bids, info.all_bids
        ))
        
        # UCB1 update: update Q-value for the chosen arm
        s = self._value_to_state(value)
        a = self.current_arm_idx
        
        # Increment counts
        self.counts[s, a] += 1
        self.total_pulls[s, a] += 1
        
        # Update Q-value using sample average
        # Q(s,a) = Q(s,a) + (1/count) * (reward - Q(s,a))
        alpha = 1.0 / self.counts[s, a]  # Learning rate decreases over time
        self.Q[s, a] = self.Q[s, a] + alpha * (utility - self.Q[s, a])


class EpsilonGreedyBanditAgent:
   
    def __init__(
        self,
        agent_id: int,
        theta_options: np.ndarray = None,
        n_value_bins: int = 10,
        epsilon: float = 0.1,
        alpha: float = 0.1  # Fixed learning rate
    ):
        self.agent_id = agent_id
        
        if theta_options is None:
            theta_options = np.linspace(0.0, 1.0, 1000)  # Match QLearningAgent (1000 options)
        self.theta_options = theta_options
        self.n_arms = len(theta_options)
        
        self.n_value_bins = n_value_bins
        self.epsilon = epsilon
        self.alpha = alpha
        
        # Q-values per (state, arm)
        self.Q = np.zeros((n_value_bins, self.n_arms))
        
        self.current_value = None
        self.current_state_idx = None
        self.current_arm_idx = None
        
        self.history = []
    
    def _value_to_state(self, v: float) -> int:
        idx = int(v * self.n_value_bins)
        if idx >= self.n_value_bins:
            idx = self.n_value_bins - 1
        return idx
    
    def draw_value(self) -> float:
        v = np.random.uniform(0.0, 1.0)
        self.current_value = v
        self.current_state_idx = self._value_to_state(v)
        return v
    
    def choose_theta(self) -> float:
        assert self.current_state_idx is not None, "call draw_value() before choose_theta()"
        s = self.current_state_idx
        
        if np.random.rand() < self.epsilon:
            # Explore: random arm
            arm_idx = np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            best_arms = np.where(self.Q[s] == self.Q[s].max())[0]
            arm_idx = np.random.choice(best_arms)
        
        self.current_arm_idx = arm_idx
        return self.theta_options[arm_idx]
    
    def compute_utility(self, value: float, won: bool, price_paid: float) -> float:
        return (value - price_paid) if won else 0.0
    
    def should_dropout_ascending(self, current_price: float, value: float, max_willingness: float) -> bool:
        """Decide whether to drop out in ascending clock auction"""
        return current_price >= max_willingness
    
    def should_dropout_descending(self, current_price: float, value: float, min_willingness: float) -> bool:
        """Decide whether to drop out in descending clock auction"""
        return current_price <= min_willingness
    
    def update(self, value: float, chosen_theta: float, outcome: AuctionOutcome):
        info = outcome.agent_info[self.agent_id]
        bid = value * chosen_theta
        won = info.won
        utility = self.compute_utility(value, won, outcome.winning_bid)
        
        self.history.append((
            value, bid, utility, chosen_theta,
            info.winning_bid, info.losing_bids, info.all_bids
        ))
        
        s = self._value_to_state(value)
        a = self.current_arm_idx
        
        # Q-learning style update with fixed learning rate
        self.Q[s, a] = self.Q[s, a] + self.alpha * (utility - self.Q[s, a])


def plot_imperfect_info_results(
    n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist,
    info_type: InformationType, collusion_metrics=None, auction_type: AuctionType = None, agent_type: str = None
):
    """Plot results for imperfect information experiments"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    title_suffix = f"{auction_type.value if auction_type else ''} - {info_type.value}"
    
    # Theta convergence
    ax = axes[0, 0]
    for i, hist in enumerate(theta_hist):
        ax.plot(hist, alpha=0.3, label=f'agent {i}' if i < 3 else '')
    ax.axhline(y=(n_agents-1)/n_agents, color='r', linestyle='--', 
               label=f'theory (n-1)/n={(n_agents-1)/n_agents:.2f}')
    ax.plot(avg_theta_hist, color='black', linewidth=2, label='avg theta')
    ax.set_xlabel('round')
    ax.set_ylabel('theta (shading factor)')
    ax.set_title(f'theta convergence - {title_suffix}')
    ax.legend()
    
    # Efficiency over time
    ax = axes[0, 1]
    window = 100
    efficiency_smooth = np.convolve(efficiency_hist, np.ones(window)/window, mode='valid')
    ax.plot(efficiency_smooth)
    ax.axhline(y=1.0, color='r', linestyle='--', label='perfect efficiency')
    ax.set_xlabel('round')
    ax.set_ylabel('fraction efficient (rolling avg)')
    ax.set_title('auction efficiency')
    ax.legend()
    
    # Revenue over time
    ax = axes[0, 2]
    revenue_smooth = np.convolve(revenue_hist, np.ones(window)/window, mode='valid')
    ax.plot(revenue_smooth)
    ax.set_xlabel('round')
    ax.set_ylabel('avg revenue')
    ax.set_title('auctioneer revenue')
    
    # Final theta distribution
    ax = axes[1, 0]
    final_thetas = [hist[-1] for hist in theta_hist]
    ax.hist(final_thetas, bins=20, alpha=0.7)
    ax.axvline(x=(n_agents-1)/n_agents, color='r', linestyle='--', label='theory')
    ax.set_xlabel('final theta')
    ax.set_ylabel('count')
    ax.set_title('final theta distribution')
    ax.legend()
    
    # Bid variance over time (collusion indicator)
    ax = axes[1, 1]
    if collusion_metrics and 'bid_variance' in collusion_metrics:
        variance_smooth = np.convolve(collusion_metrics['bid_variance'], 
                                      np.ones(window)/window, mode='valid')
        ax.plot(variance_smooth)
        ax.set_xlabel('round')
        ax.set_ylabel('bid variance (rolling avg)')
        ax.set_title('bid variance (low = potential collusion)')
    
    # Win rate distribution
    ax = axes[1, 2]
    if collusion_metrics and 'win_rates' in collusion_metrics:
        final_win_rates = collusion_metrics['win_rates'][-1] if len(collusion_metrics['win_rates']) > 0 else []
        if len(final_win_rates) > 0:
            ax.bar(range(len(final_win_rates)), final_win_rates)
            ax.axhline(y=1.0/n_agents, color='r', linestyle='--', label='fair (1/n)')
            ax.set_xlabel('agent')
            ax.set_ylabel('win rate')
            ax.set_title('final win rate distribution')
            ax.legend()
    
    plt.tight_layout()
    agent_suffix = f"_{agent_type}" if agent_type else ""
    filename = f'graphs/auction_{auction_type.value if auction_type else "first_price"}_{info_type.value}{agent_suffix}.png'
    plt.savefig(filename, dpi=150)
    plt.close()  # Close to avoid showing plot during batch run


def detect_collusion(agents, window: int = 1000):
    """
    Detect potential collusion behavior.
    
    Indicators:
    - Low bid variance (agents coordinating bids)
    - Uneven win rates (some agents winning more than fair share)
    - Similar theta values across agents
    """
    if len(agents) == 0 or len(agents[0].history) < window:
        return {}
    
    # Extract recent bids
    recent_bids = []
    for agent in agents:
        recent_history = agent.history[-window:]
        recent_bids.append([h[1] for h in recent_history])  # bids are at index 1
    
    # Bid variance
    all_recent_bids = np.concatenate(recent_bids)
    bid_variance = np.var(all_recent_bids)
    
    # Win rates
    win_rates = []
    for agent in agents:
        recent_history = agent.history[-window:]
        wins = sum(1 for h in recent_history if h[2] > 0)  # utility > 0 means won
        win_rates.append(wins / len(recent_history))
    
    # Theta similarity (low variance suggests coordination)
    recent_thetas = []
    for agent in agents:
        recent_history = agent.history[-window:]
        recent_thetas.append([h[3] for h in recent_history])  # theta at index 3
    
    avg_thetas = [np.mean(t) for t in recent_thetas]
    theta_variance = np.var(avg_thetas)
    
    return {
        'bid_variance': bid_variance,
        'win_rates': win_rates,
        'theta_variance': theta_variance,
        'collusion_score': 1.0 / (1.0 + bid_variance + theta_variance)  # Higher = more suspicious
    }


def run_imperfect_info_simulation(
    n_agents: int,
    n_rounds: int,
    info_type: InformationType,
    auction_type: AuctionType = AuctionType.FIRST_PRICE,
    agent_type: str = "ucb",  # "ucb" or "epsilon_greedy"
    theta_options: np.ndarray = None
):
    """
    Run auction simulation with imperfect information and bandit agents.
    
    Args:
        n_agents: Number of agents
        n_rounds: Number of auction rounds
        info_type: Type of information revelation
        auction_type: Type of auction mechanism (FIRST_PRICE, ASCENDING_CLOCK, DESCENDING_CLOCK)
        agent_type: "ucb" for UCB1 or "epsilon_greedy" for epsilon-greedy
        theta_options: Discretized theta space (optional)
    """
    # Create auction mechanism
    if auction_type == AuctionType.FIRST_PRICE:
        auction = FirstPriceAuction(n_agents, info_type)
    elif auction_type == AuctionType.ASCENDING_CLOCK:
        auction = AscendingClockAuction(n_agents, info_type)
    elif auction_type == AuctionType.DESCENDING_CLOCK:
        auction = DescendingClockAuction(n_agents, info_type)
    else:
        raise ValueError(f"Unknown auction_type: {auction_type}")
    
    # Create bandit agents
    if agent_type == "ucb":
        agents = [BanditAgent(i, theta_options=theta_options) for i in range(n_agents)]
    elif agent_type == "epsilon_greedy":
        agents = [EpsilonGreedyBanditAgent(i, theta_options=theta_options) for i in range(n_agents)]
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    
    # Tracking
    theta_hist = [[] for _ in range(n_agents)]
    avg_theta_hist = []
    efficiency_hist = []
    revenue_hist = []
    collusion_metrics_hist = []
    
    for round_idx in range(n_rounds):
        # Each agent draws value and chooses theta
        values = np.array([agent.draw_value() for agent in agents])
        thetas = [agent.choose_theta() for agent in agents] 
        
        # Run auction based on type
        if auction_type == AuctionType.FIRST_PRICE:
            bids = np.array([values[i] * thetas[i] for i in range(n_agents)])
            outcome = auction.run_auction(bids, values)
        elif auction_type == AuctionType.ASCENDING_CLOCK:
            # For ascending clock: theta determines max willingness to pay
            max_willingness = np.array([values[i] * thetas[i] for i in range(n_agents)])
            
            def dropout_decisions(price, active_agents, values, max_willingness):
                decisions = np.zeros(n_agents, dtype=bool)
                for agent_idx in active_agents:
                    decisions[agent_idx] = agents[agent_idx].should_dropout_ascending(
                        price, values[agent_idx], max_willingness[agent_idx]
                    )
                return decisions
            
            outcome = auction.run_auction(dropout_decisions, values, max_willingness)
        elif auction_type == AuctionType.DESCENDING_CLOCK:
            # For descending clock: theta determines min willingness to pay
            min_willingness = np.array([values[i] * thetas[i] for i in range(n_agents)])
            
            def dropout_decisions(price, active_agents, values, min_willingness):
                decisions = np.zeros(n_agents, dtype=bool)
                for agent_idx in active_agents:
                    decisions[agent_idx] = agents[agent_idx].should_dropout_descending(
                        price, values[agent_idx], min_willingness[agent_idx]
                    )
                return decisions
            
            outcome = auction.run_auction(dropout_decisions, values, min_willingness)
        
        # Agents update based on revealed information
        for i, agent in enumerate(agents):
            agent.update(values[i], thetas[i], outcome)
        
        # Track metrics
        for i in range(n_agents):
            theta_hist[i].append(thetas[i])
        
        avg_theta_hist.append(np.mean(thetas))
        
        # Efficiency: did highest-value agent win?
        highest_value_idx = np.argmax(values)
        efficiency_hist.append(1.0 if outcome.winner_idx == highest_value_idx else 0.0)
        
        # Revenue
        revenue_hist.append(outcome.winning_bid)
        
        # Collusion detection (every 100 rounds)
        if round_idx % 100 == 0 and round_idx > 0:
            collusion_metrics = detect_collusion(agents, window=min(1000, round_idx))
            if collusion_metrics:
                collusion_metrics_hist.append({
                    'round': round_idx,
                    'bid_variance': collusion_metrics['bid_variance'],
                    'win_rates': collusion_metrics['win_rates'],
                    'theta_variance': collusion_metrics['theta_variance'],
                    'collusion_score': collusion_metrics['collusion_score']
                })
        
        # Logging
        if round_idx % 5000 == 0:
            avg_theta = np.mean([thetas[i] for i in range(n_agents)])
            print(f"round {round_idx}: avg_theta={avg_theta:.3f}, "
                  f"theory={(n_agents-1)/n_agents:.3f}, "
                  f"auction={auction_type.value}, info={info_type.value}")
    
    # Final diagnostics
    print(f"\n=== Final Results ({auction_type.value}, {info_type.value}) ===")
    print(f"Theory predicts theta = {(n_agents-1)/n_agents:.3f}")
    final_thetas = [hist[-1] for hist in theta_hist]
    print(f"Learned avg theta = {np.mean(final_thetas):.3f}")
    print(f"Final efficiency = {np.mean(efficiency_hist[-1000:]):.3f}")
    print(f"Avg revenue (last 1000) = {np.mean(revenue_hist[-1000:]):.3f}")
    
    # Collusion analysis
    if collusion_metrics_hist:
        final_collusion = collusion_metrics_hist[-1]
        print("\nCollusion indicators:")
        print(f"  Bid variance: {final_collusion['bid_variance']:.4f}")
        print(f"  Theta variance: {final_collusion['theta_variance']:.4f}")
        print(f"  Collusion score: {final_collusion['collusion_score']:.4f}")
        print(f"  Win rates: {[f'{wr:.3f}' for wr in final_collusion['win_rates']]}")
    
    # Prepare collusion metrics for plotting
    collusion_plot_data = {}
    if collusion_metrics_hist:
        collusion_plot_data['bid_variance'] = [m['bid_variance'] for m in collusion_metrics_hist]
        collusion_plot_data['win_rates'] = [m['win_rates'] for m in collusion_metrics_hist]
    
    # Plot results
    plot_imperfect_info_results(
        n_agents, theta_hist, avg_theta_hist, efficiency_hist, revenue_hist,
        info_type, collusion_plot_data, auction_type, agent_type
    )
    
    return agents, {
        'theta_hist': theta_hist,
        'efficiency_hist': efficiency_hist,
        'revenue_hist': revenue_hist,
        'collusion_metrics': collusion_metrics_hist
    }


if __name__ == "__main__":
    # Configuration
    n_agents = 10
    n_rounds = 50000
    theta_options = np.linspace(0.0, 1.0, 1000)  # Match QLearningAgent (1000 options)
    
    # Options:
    # - "single": Run just one first-price auction with minimal info
    # - "first_price_all": Run first-price auctions for all info types
    # - "all": Run all auction types and info types
    RUN_MODE = "single"  # Change to "first_price_all" or "all" for more experiments
    
    print("Running imperfect information auction experiments...")
    print("=" * 60)
    
    results = {}
    
    if RUN_MODE == "single":
        # Run just one: first-price with minimal information
        print("\n" + "="*60)
        print("Auction Type: FIRST_PRICE")
        print("Information Type: MINIMAL")
        print("="*60)
        agents, metrics = run_imperfect_info_simulation(
            n_agents=n_agents,
            n_rounds=n_rounds,
            info_type=InformationType.MINIMAL,
            auction_type=AuctionType.FIRST_PRICE,
            agent_type="ucb",
            theta_options=theta_options
        )
        results[(AuctionType.FIRST_PRICE, InformationType.MINIMAL)] = {'agents': agents, 'metrics': metrics}
    
    elif RUN_MODE == "first_price_all":
        # Run first-price auctions for all information types
        auction_type = AuctionType.FIRST_PRICE
        for info_type in InformationType:
            print("\n" + "="*60)
            print(f"Auction Type: {auction_type.value.upper()}")
            print(f"Information Type: {info_type.value.upper()}")
            print("="*60)
            agents, metrics = run_imperfect_info_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                info_type=info_type,
                auction_type=auction_type,
                agent_type="ucb",
                theta_options=theta_options
            )
            results[(auction_type, info_type)] = {'agents': agents, 'metrics': metrics}
    
    elif RUN_MODE == "all":
        # Run for each auction type and information type combination
        for auction_type in AuctionType:
            for info_type in InformationType:
                print("\n" + "="*60)
                print(f"Auction Type: {auction_type.value.upper()}")
                print(f"Information Type: {info_type.value.upper()}")
                print("="*60)
                agents, metrics = run_imperfect_info_simulation(
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    info_type=info_type,
                    auction_type=auction_type,
                    agent_type="ucb",
                    theta_options=theta_options
                )
                results[(auction_type, info_type)] = {'agents': agents, 'metrics': metrics}
    
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}. Use 'single', 'first_price_all', or 'all'")

