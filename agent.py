import numpy as np

class Agent:
    def __init__(self, agent_id: int, learning_rate: float = 0.01):
        self.agent_id = agent_id
        self.theta = np.random.uniform(0, 0.5)  # start with some shading
        self.lr = learning_rate
        self.history = []  # track (v, bid, utility, theta) for analysis
        
    def draw_value(self) -> float:
        """draw private value for this auction"""
        return np.random.uniform(0, 1)
    
    def compute_bid(self, value: float) -> float:
        """bid = v * (1 - theta)"""
        return value * self.theta
    
    def compute_utility(self, value: float, won: bool, price_paid: float) -> float:
        """utility = v - price if won, else 0"""
        return (value - price_paid) if won else 0.0
    
    def update(self, value: float, bid: float, utility: float):
        """policy gradient update on theta"""
        # gradient of log(policy) w.r.t theta
        # policy is deterministic bid = v(1-theta), so we use score function
        # simplified: gradient points toward increasing utility
        
        # if utility > 0 (won profitably): could shade more (increase theta)
        # if utility = 0 (lost): should shade less (decrease theta)
        # this is rough heuristic, you'll tune this
        
        if utility > 0:
            # won - consider shading more next time
            gradient = value  # direction to increase theta
            self.theta += self.lr * utility * (gradient / (value + 1e-8))
        else:
            # lost - shade less
            gradient = -value
            self.theta -= self.lr * 0.1  # small nudge toward bidding higher
        
        # keep theta in reasonable bounds
        self.theta = np.clip(self.theta, 0, 0.99)
        
        self.history.append((value, bid, utility, self.theta))
