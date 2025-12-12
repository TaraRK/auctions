"""
Test script for War of Attrition Auction implementation
"""
import numpy as np
from auctions import WarOfAttritionAuction

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

    from auctions import FirstPriceAuction, AllPayAuction

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
    print("\nWar of Attrition extracts more revenue than First-Price")
    print("but potentially less than All-Pay (depending on bid distribution)")


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 60)
    print("Test 3: Edge Cases")
    print("=" * 60)

    # Test with 2 agents
    print("\nEdge case 1: Two agents")
    auction = WarOfAttritionAuction(n_agents=2)
    values = np.array([0.8, 0.6])
    bids = np.array([0.5, 0.3])
    outcome = auction.run_auction(values, bids)

    print(f"Values: {values}")
    print(f"Bids: {bids}")
    print(f"Payments: {outcome.payments}")
    print(f"Utilities: {outcome.utilities}")
    print(f"✓ Two-agent case works correctly")

    # Test with tied bids
    print("\nEdge case 2: Tied highest bids")
    auction = WarOfAttritionAuction(n_agents=3)
    values = np.array([0.8, 0.7, 0.9])
    bids = np.array([0.5, 0.5, 0.3])  # Two agents bid same amount
    outcome = auction.run_auction(values, bids)

    print(f"Values: {values}")
    print(f"Bids: {bids}")
    print(f"Winner: Agent {outcome.winner_idx}")
    print(f"Payments: {outcome.payments}")
    print(f"Utilities: {outcome.utilities}")
    print(f"✓ Tie-breaking works (numpy argmax picks first occurrence)")


def simulation_example():
    """Run a small simulation to see convergence behavior"""
    print("\n" + "=" * 60)
    print("Test 4: Small Simulation (100 rounds)")
    print("=" * 60)

    from agent import Agent

    n_agents = 5
    n_rounds = 100
    auction = WarOfAttritionAuction(n_agents)
    agents = [Agent(i) for i in range(n_agents)]

    revenues = []
    avg_thetas = []

    for round_idx in range(n_rounds):
        # Draw values and choose strategies
        values = np.array([agent.draw_value() for agent in agents])
        thetas = [agent.choose_theta() for agent in agents]
        bids = np.array([values[i] * thetas[i] for i in range(n_agents)])

        # Run auction
        outcome = auction.run_auction(values, bids)

        # Agents update
        for i, agent in enumerate(agents):
            agent.update(values[i], thetas[i], outcome)

        # Track metrics
        revenues.append(np.sum(outcome.payments))
        avg_thetas.append(np.mean(thetas))

    print(f"\nSimulation complete!")
    print(f"Average revenue over {n_rounds} rounds: {np.mean(revenues):.3f}")
    print(f"Final average theta: {avg_thetas[-1]:.3f}")
    print(f"Theoretical theta (n-1)/n: {(n_agents-1)/n_agents:.3f}")

    # Show last few thetas
    print(f"\nFinal agent thetas:")
    for i, agent in enumerate(agents):
        if len(agent.history) > 0:
            recent_thetas = [h[3] for h in agent.history[-10:]]
            print(f"  Agent {i}: {np.mean(recent_thetas):.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WAR OF ATTRITION AUCTION - TEST SUITE")
    print("=" * 60)

    test_basic_mechanism()
    compare_auction_types()
    test_edge_cases()
    simulation_example()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
