"""
Analyze the learned auction mechanism from AMD simulation
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from amd_bandits import AllocationNetwork, PaymentNetwork, LearningAuctioneer
from typing import Dict, List, Tuple

def analyze_allocation_rule(auctioneer: LearningAuctioneer, n_agents: int = 10) -> Dict:
    """
    Analyze what allocation rule the G network learned
    
    Tests:
    1. Does it allocate to highest bidder?
    2. How does it respond to bid differences?
    3. Is it deterministic or probabilistic?
    """
    print("\n" + "="*80)
    print("ANALYZING LEARNED ALLOCATION RULE (G Network)")
    print("="*80)
    
    results = {
        'highest_bidder_allocation': [],
        'bid_difference_sensitivity': [],
        'allocation_probabilities': []
    }
    
    # Test 1: Does it allocate to highest bidder?
    print("\n1. Testing allocation to highest bidder:")
    print("-" * 80)
    n_tests = 100
    correct_allocations = 0
    
    for _ in range(n_tests):
        # Generate random bids
        bids = np.random.uniform(0, 1, n_agents)
        highest_bidder = np.argmax(bids)
        
        # Get allocation
        winner_idx, allocation_probs = auctioneer.G_network.allocate(bids, temperature=1.0)
        
        if winner_idx == highest_bidder:
            correct_allocations += 1
        
        results['highest_bidder_allocation'].append({
            'bids': bids.copy(),
            'highest_bidder': highest_bidder,
            'allocated_to': winner_idx,
            'allocation_probs': allocation_probs.copy()
        })
    
    accuracy = correct_allocations / n_tests
    print(f"   Allocated to highest bidder: {correct_allocations}/{n_tests} ({accuracy*100:.1f}%)")
    
    # Test 2: Sensitivity to bid differences
    print("\n2. Testing sensitivity to bid differences:")
    print("-" * 80)
    bid_differences = [0.01, 0.05, 0.10, 0.20, 0.50]
    
    for diff in bid_differences:
        # Create scenario: agent 0 bids high, others bid low
        bids = np.ones(n_agents) * 0.3
        bids[0] = 0.3 + diff
        
        winner_idx, allocation_probs = auctioneer.G_network.allocate(bids, temperature=1.0)
        prob_agent_0 = allocation_probs[0]
        
        results['bid_difference_sensitivity'].append({
            'bid_difference': diff,
            'prob_agent_0': prob_agent_0,
            'allocated_to_agent_0': (winner_idx == 0)
        })
        
        print(f"   Bid difference: {diff:.2f} → Agent 0 prob: {prob_agent_0:.3f}, "
              f"Winner: Agent {winner_idx}")
    
    # Test 3: Allocation probabilities distribution
    print("\n3. Allocation probability distribution:")
    print("-" * 80)
    all_probs = []
    for result in results['highest_bidder_allocation']:
        all_probs.extend(result['allocation_probs'])
    
    all_probs = np.array(all_probs)
    print(f"   Mean allocation prob: {np.mean(all_probs):.4f}")
    print(f"   Std allocation prob: {np.std(all_probs):.4f}")
    print(f"   Min allocation prob: {np.min(all_probs):.4f}")
    print(f"   Max allocation prob: {np.max(all_probs):.4f}")
    
    results['allocation_probabilities'] = all_probs
    
    return results


def analyze_payment_rule(auctioneer: LearningAuctioneer, n_agents: int = 10) -> Dict:
    """
    Analyze what payment rule the P network learned
    
    Tests:
    1. How does payment relate to bid?
    2. Does it charge winner or all agents?
    3. Payment as function of bid amount
    """
    print("\n" + "="*80)
    print("ANALYZING LEARNED PAYMENT RULE (P Network)")
    print("="*80)
    
    results = {
        'payment_vs_bid': [],
        'payment_fraction': [],
        'loser_payments': []
    }
    
    # Test 1: Payment vs bid relationship
    print("\n1. Testing payment as function of winning bid:")
    print("-" * 80)
    print(f"{'Winning Bid':<15} {'Payment':<15} {'Payment/Bid':<15} {'Max Allowed':<15}")
    print("-" * 80)
    
    test_bids = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for winning_bid in test_bids:
        # Create scenario: agent 0 has winning bid
        bids = np.ones(n_agents) * 0.1
        bids[0] = winning_bid
        winner_idx = 0
        
        # Get allocation
        _, allocation_probs = auctioneer.G_network.allocate(bids, temperature=1.0)
        
        # Get payments
        payments = auctioneer.P_network.compute_payments(bids, winner_idx, allocation_probs)
        payment = payments[winner_idx]
        max_allowed = winning_bid * 1.5
        
        payment_fraction = payment / winning_bid if winning_bid > 0 else 0
        
        results['payment_vs_bid'].append({
            'winning_bid': winning_bid,
            'payment': payment,
            'payment_fraction': payment_fraction,
            'max_allowed': max_allowed
        })
        
        print(f"{winning_bid:<15.3f} {payment:<15.3f} {payment_fraction:<15.3f} {max_allowed:<15.3f}")
    
    # Test 2: Payment fraction distribution
    print("\n2. Payment fraction (payment/bid) distribution:")
    print("-" * 80)
    payment_fractions = [r['payment_fraction'] for r in results['payment_vs_bid']]
    print(f"   Mean payment fraction: {np.mean(payment_fractions):.4f}")
    print(f"   Std payment fraction: {np.std(payment_fractions):.4f}")
    print(f"   Min payment fraction: {np.min(payment_fractions):.4f}")
    print(f"   Max payment fraction: {np.max(payment_fractions):.4f}")
    
    results['payment_fraction'] = payment_fractions
    
    # Test 3: Do losers pay?
    print("\n3. Testing if losers pay:")
    print("-" * 80)
    bids = np.random.uniform(0, 1, n_agents)
    winner_idx = np.argmax(bids)
    _, allocation_probs = auctioneer.G_network.allocate(bids, temperature=1.0)
    payments = auctioneer.P_network.compute_payments(bids, winner_idx, allocation_probs)
    
    loser_payments = [payments[i] for i in range(n_agents) if i != winner_idx]
    results['loser_payments'] = loser_payments
    
    print(f"   Winner payment: {payments[winner_idx]:.4f}")
    print(f"   Loser payments: mean={np.mean(loser_payments):.4f}, "
          f"max={np.max(loser_payments):.4f}, "
          f"non-zero={np.sum(np.array(loser_payments) > 1e-6)}/{len(loser_payments)}")
    
    return results


def compare_to_standard_mechanisms(auctioneer: LearningAuctioneer, n_agents: int = 10) -> Dict:
    """
    Compare learned mechanism to standard auction formats
    """
    print("\n" + "="*80)
    print("COMPARING TO STANDARD MECHANISMS")
    print("="*80)
    
    # Generate test scenarios
    n_scenarios = 100
    scenarios = []
    
    for _ in range(n_scenarios):
        values = np.random.uniform(0, 1, n_agents)
        # Agents bid with theta = 0.55 (observed average)
        bids = values * 0.55
        scenarios.append({
            'values': values.copy(),
            'bids': bids.copy()
        })
    
    # Test learned mechanism
    learned_revenues = []
    learned_efficiency = []
    
    for scenario in scenarios:
        bids = scenario['bids']
        values = scenario['values']
        
        winner_idx, allocation_probs = auctioneer.G_network.allocate(bids, temperature=1.0)
        payments = auctioneer.P_network.compute_payments(bids, winner_idx, allocation_probs)
        revenue = payments[winner_idx]
        
        learned_revenues.append(revenue)
        learned_efficiency.append(1.0 if winner_idx == np.argmax(values) else 0.0)
    
    # Test first-price (theoretical)
    first_price_revenues = []
    first_price_efficiency = []
    
    for scenario in scenarios:
        bids = scenario['bids']
        values = scenario['values']
        
        winner_idx = np.argmax(bids)
        revenue = bids[winner_idx]
        
        first_price_revenues.append(revenue)
        first_price_efficiency.append(1.0 if winner_idx == np.argmax(values) else 0.0)
    
    # Test second-price (theoretical)
    second_price_revenues = []
    second_price_efficiency = []
    
    for scenario in scenarios:
        bids = scenario['bids']
        values = scenario['values']
        
        winner_idx = np.argmax(bids)
        sorted_bids = np.sort(bids)
        revenue = sorted_bids[-2] if len(sorted_bids) > 1 else sorted_bids[-1]
        
        second_price_revenues.append(revenue)
        second_price_efficiency.append(1.0 if winner_idx == np.argmax(values) else 0.0)
    
    # Compare
    print("\nRevenue Comparison:")
    print("-" * 80)
    print(f"Learned mechanism:  {np.mean(learned_revenues):.4f} ± {np.std(learned_revenues):.4f}")
    print(f"First-price:       {np.mean(first_price_revenues):.4f} ± {np.std(first_price_revenues):.4f}")
    print(f"Second-price:       {np.mean(second_price_revenues):.4f} ± {np.std(second_price_revenues):.4f}")
    
    print("\nEfficiency Comparison:")
    print("-" * 80)
    print(f"Learned mechanism:  {np.mean(learned_efficiency):.4f}")
    print(f"First-price:       {np.mean(first_price_efficiency):.4f}")
    print(f"Second-price:       {np.mean(second_price_efficiency):.4f}")
    
    return {
        'learned_revenue': np.mean(learned_revenues),
        'first_price_revenue': np.mean(first_price_revenues),
        'second_price_revenue': np.mean(second_price_revenues),
        'learned_efficiency': np.mean(learned_efficiency),
        'first_price_efficiency': np.mean(first_price_efficiency),
        'second_price_efficiency': np.mean(second_price_efficiency)
    }


def summarize_learned_mechanism(auctioneer: LearningAuctioneer, n_agents: int = 10):
    """
    Provide a summary of what mechanism was learned
    """
    print("\n" + "="*80)
    print("SUMMARY: LEARNED AUCTION MECHANISM")
    print("="*80)
    
    # Analyze allocation
    alloc_results = analyze_allocation_rule(auctioneer, n_agents)
    
    # Analyze payment
    payment_results = analyze_payment_rule(auctioneer, n_agents)
    
    # Compare to standards
    comparison = compare_to_standard_mechanisms(auctioneer, n_agents)
    
    # Summary
    print("\n" + "="*80)
    print("MECHANISM SUMMARY")
    print("="*80)
    
    alloc_accuracy = sum(1 for r in alloc_results['highest_bidder_allocation'] 
                        if r['allocated_to'] == r['highest_bidder']) / len(alloc_results['highest_bidder_allocation'])
    
    avg_payment_fraction = np.mean(payment_results['payment_fraction'])
    
    print(f"\nAllocation Rule:")
    print(f"  - Allocates to highest bidder: {alloc_accuracy*100:.1f}% of the time")
    print(f"  - Uses softmax with temperature (probabilistic allocation)")
    
    print(f"\nPayment Rule:")
    print(f"  - Charges winner: {avg_payment_fraction*100:.1f}% of winning bid (on average)")
    print(f"  - Payment constraint: ≤ 1.5 × max_bid")
    print(f"  - Losers pay: {np.sum(np.array(payment_results['loser_payments']) > 1e-6)}/{len(payment_results['loser_payments'])} pay non-zero amounts")
    
    print(f"\nPerformance:")
    print(f"  - Revenue: {comparison['learned_revenue']:.4f} (vs first-price: {comparison['first_price_revenue']:.4f})")
    print(f"  - Efficiency: {comparison['learned_efficiency']:.2%} (vs first-price: {comparison['first_price_efficiency']:.2%})")
    
    print(f"\nInterpretation:")
    if alloc_accuracy > 0.8:
        print("  - Allocation: Learned to allocate to highest bidder (similar to standard auctions)")
    else:
        print("  - Allocation: Does NOT consistently allocate to highest bidder (suboptimal)")
    
    if 0.8 <= avg_payment_fraction <= 1.0:
        print("  - Payment: Learned first-price-like payment (charges close to bid)")
    elif avg_payment_fraction < 0.5:
        print("  - Payment: Learned to charge LESS than bid (revenue-reducing)")
    else:
        print(f"  - Payment: Learned intermediate payment rule ({avg_payment_fraction:.2%} of bid)")
    
    return {
        'allocation_accuracy': alloc_accuracy,
        'avg_payment_fraction': avg_payment_fraction,
        'comparison': comparison
    }


if __name__ == "__main__":
    # This would be called after running the AMD simulation
    # Example usage:
    # from amd_bandits import run_amd_simulation, InformationType
    # agents, auctioneer, metrics = run_amd_simulation(...)
    # summarize_learned_mechanism(auctioneer, n_agents=10)
    
    print("This script analyzes the learned mechanism from AMD simulation.")
    print("Run it after completing the AMD simulation to see what mechanism was learned.")
    pass

