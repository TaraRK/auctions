
class WarOfAttritionAuction:
    """
    woa 
    """
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def run_auction(self, values: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
        
        winner_idx = np.argmax(bids)
        winning_bid = bids[winner_idx]

        
        second_highest_bid = np.sort(bids)[-2] if self.n_agents > 1 else bids[winner_idx]

        
        payments = np.minimum(bids, second_highest_bid)

        utilities = -payments  
        utilities[winner_idx] = values[winner_idx] - payments[winner_idx]  # Winner also gets value

        return AuctionOutcome(
            winner_idx=winner_idx,
            winning_bid=winning_bid,
            all_bids=bids.copy(),
            payments=payments,
            utilities=utilities,
        )
