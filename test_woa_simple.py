"""
Simple test for War of Attrition Auction implementation
"""
import numpy as np
from auctions import WarOfAttritionAuction, FirstPriceAuction, AllPayAuction

def test_basic_mechanism():
    """Test basic war of attrition mechanism with simple example"""
    print("=" * 60)
    print("Test 1: Basic War of Attrition Mechanism")
    print("=" * 60)

    auction = WarOfAttritionAuction(n_agents=3)

    # Example: 3 agents with values [0.8, 0.6, 0.9]
    values = np.array([0.8, 0.6, 0.9])
    bids = np.array([0.5, 0.3, 0.7])  # Agent 2 bids highest

    print(f"Values: {values}")
    print(f"Bids: {bids}")

    outcome = auction.run_auction(values, bids)

    print(f"\nWinner: Agent {outcome.winner_idx}")
    print(f"Winning bid: {outcome.winning_bid}")
    print(f"Second-highest bid (what everyone pays): {np.sort(bids)[-2]}")
    print(f"\nPayments: {outcome.payments}")
    print(f"Utilities: {outcome.utilities}")

    # Verify payment rule
    second_highest = np.sort(bids)[-2]
    assert np.all(outcome.payments == second_highest), "All agents should pay second-highest bid"

    # Verify winner utility
    expected_winner_utility = values[outcome.winner_idx] - second_highest
    assert np.isclose(outcome.utilities[outcome.winner_idx], expected_winner_utility), \
        "Winner utility should be value - second_highest_bid"

    # Verify loser utilities
    for i in range(len(values)):
        if i != outcome.winner_idx:
            assert np.isclose(outcome.utilities[i], -second_highest), \
                f"Loser {i} utility should be -second_highest_bid"

    print("\n✓ All assertions passed!")
    return outcome


def compare_auction_types():
    """Compare War of Attrition with All-Pay and First-Price auctions"""
    print("\n" + "=" * 60)
    print("Test 2: Comparing Auction Types")
    print("=" * 60)

    n_agents = 3
    values = np.array([0.8, 0.6, 0.9])
    bids = np.array([0.5, 0.3, 0.7])

    print(f"Setup: {n_agents} agents")
    print(f"Values: {values}")
    print(f"Bids: {bids}")
    print(f"Winner: Agent {np.argmax(bids)} (highest bid = {np.max(bids)})")
    print(f"Second-highest bid: {np.sort(bids)[-2]}")

    # Run all three auction types
    woa_auction = WarOfAttritionAuction(n_agents)
    fp_auction = FirstPriceAuction(n_agents)
    ap_auction = AllPayAuction(n_agents)

    woa_outcome = woa_auction.run_auction(values, bids)
    fp_outcome = fp_auction.run_auction(values, bids)
    ap_outcome = ap_auction.run_auction(values, bids)

    print("\n" + "-" * 60)
    print("FIRST-PRICE AUCTION:")
    print(f"  Payments: {fp_outcome.payments}")
    print(f"  Utilities: {fp_outcome.utilities}")
    print(f"  Total revenue: {np.sum(fp_outcome.payments):.3f}")

    print("\n" + "-" * 60)
    print("ALL-PAY AUCTION:")
    print(f"  Payments: {ap_outcome.payments}")
    print(f"  Utilities: {ap_outcome.utilities}")
    print(f"  Total revenue: {np.sum(ap_outcome.payments):.3f}")

    print("\n" + "-" * 60)
    print("WAR OF ATTRITION:")
    print(f"  Payments: {woa_outcome.payments}")
    print(f"  Utilities: {woa_outcome.utilities}")
    print(f"  Total revenue: {np.sum(woa_outcome.payments):.3f}")

    print("\n" + "=" * 60)
    print("Key Differences:")
    print("=" * 60)
    print("• First-Price: Only winner pays (their own bid)")
    print("• All-Pay: Everyone pays their OWN bid")
    print("• War of Attrition: Everyone pays the SECOND-HIGHEST bid")
    print("\nRevenue comparison:")
    print(f"  First-Price: {np.sum(fp_outcome.payments):.3f}")
    print(f"  All-Pay: {np.sum(ap_outcome.payments):.3f}")
    print(f"  War of Attrition: {np.sum(woa_outcome.payments):.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WAR OF ATTRITION AUCTION - SIMPLE TEST")
    print("=" * 60)

    test_basic_mechanism()
    compare_auction_types()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
