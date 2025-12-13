"""
Experiment runner for AMD parameter sweeps
Varies learning rate, epochs, gamma, and information type
"""

import os
import numpy as np
from itertools import product
import json
from datetime import datetime
from amd_bandits import (
    run_amd_simulation, 
    plot_amd_convergence, 
    plot_agent_strategies,
    InformationType
)
from visualize_pg_networks import plot_pg_network_outputs, print_pg_summary

def create_exp_id(lr_auctioneer, update_epochs, gamma, info_type, training_interval):
    """Create experiment identifier from parameters"""
    info_str = info_type.value.replace('_', '')
    return (f"lr{lr_auctioneer:.0e}_epochs{update_epochs}_gamma{gamma:.2f}_"
            f"info{info_str}_interval{training_interval}").replace('.', 'p').replace('+', '')

def run_experiment_sweep(
    n_agents=10,
    n_rounds=30000,
    learning_rates=[1e-4, 5e-4],
    update_epochs_list=[4],
    gammas=[0.0, 0.99],
    info_types=[InformationType.FULL_REVELATION, InformationType.MINIMAL, InformationType.WINNER],
    training_interval=[200, 1000],  
    results_dir='experiment_results'
):
    """
    Run parameter sweep experiments
    
    Args:
        n_agents: Number of agents
        n_rounds: Number of rounds per experiment
        learning_rates: List of auctioneer learning rates to test
        update_epochs_list: List of PPO update epochs to test
        gammas: List of gamma (discount factor) values to test
        info_types: List of InformationType to test
        training_interval: Training interval (fixed for now)
        results_dir: Directory to save results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/graphs', exist_ok=True)
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    
    # Track all results
    all_results = []
    best_result = None
    best_revenue = -np.inf
    
    # Handle training_interval - convert to list if single value
    if isinstance(training_interval, (int, float)):
        training_intervals = [int(training_interval)]
    else:
        training_intervals = [int(ti) for ti in training_interval]
    
    # Generate all parameter combinations
    param_combinations = list(product(learning_rates, update_epochs_list, gammas, info_types, training_intervals))
    total_experiments = len(param_combinations)
    
    print("=" * 80)
    print("Starting Experiment Sweep")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print("Parameters:")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Update epochs: {update_epochs_list}")
    print(f"  Gammas: {gammas}")
    print(f"  Info types: {[it.value for it in info_types]}")
    print(f"  Training intervals: {training_intervals}")
    print(f"  Agents: {n_agents}, Rounds: {n_rounds}")
    print("=" * 80)
    
    for exp_idx, (lr_auctioneer, update_epochs, gamma, info_type, training_interval_val) in enumerate(param_combinations, 1):
        exp_id = create_exp_id(lr_auctioneer, update_epochs, gamma, info_type, training_interval_val)
        
        print(f"\n{'='*80}")
        print(f"Experiment {exp_idx}/{total_experiments}: {exp_id}")
        print(f"{'='*80}")
        print(f"  Learning rate: {lr_auctioneer:.0e}")
        print(f"  Update epochs: {update_epochs}")
        print(f"  Gamma: {gamma:.2f}")
        print(f"  Info type: {info_type.value}")
        print(f"  Training interval: {training_interval_val}")
        
        try:
            # Run simulation
            agents, auctioneer, metrics = run_amd_simulation(
                n_agents=n_agents,
                n_rounds=n_rounds,
                info_type=info_type,
                agent_type="ppo",
                theta_options=None,
                alternate_training=True,
                training_interval=training_interval_val,
                lr_auctioneer=lr_auctioneer,
                update_epochs=update_epochs,
                gamma=gamma
            )
            
            # Extract results
            result = {
                'exp_id': exp_id,
                'parameters': {
                    'lr_auctioneer': float(lr_auctioneer),
                    'update_epochs': int(update_epochs),
                    'gamma': float(gamma),
                    'info_type': info_type.value,
                    'training_interval': int(training_interval_val),
                    'n_agents': int(n_agents),
                    'n_rounds': int(n_rounds)
                },
                'results': {
                    'final_avg_theta': float(metrics['final_avg_theta']),
                    'final_efficiency': float(metrics['final_efficiency']),
                    'final_revenue': float(metrics['final_revenue'])
                },
                'metrics': {
                    'theta_hist': [[float(x) for x in hist[-1000:]] for hist in metrics['theta_hist']],  # Last 1000 only
                    'revenue_hist': [float(x) for x in metrics['revenue_hist'][-1000:]],
                    'efficiency_hist': [float(x) for x in metrics['efficiency_hist'][-1000:]]
                }
            }
            
            # Add P&G outputs summary (last 100 rounds for JSON serialization)
            if 'pg_outputs' in metrics and len(metrics['pg_outputs']) > 0:
                pg_sample = metrics['pg_outputs'][-100:]  # Last 100 rounds
                result['pg_outputs_summary'] = [
                    {
                        'round': int(out['round']),
                        'winner_idx': int(out['winner_idx']),
                        'winning_bid': float(out['winning_bid']),
                        'winning_payment': float(out['winning_payment']),
                        'payment_ratio': float(out['payment_ratio']),
                        'revenue': float(out['revenue'])
                    }
                    for out in pg_sample
                ]
            
            # Check if this is the best result (by revenue)
            if metrics['final_revenue'] > best_revenue:
                best_revenue = metrics['final_revenue']
                best_result = result.copy()
            
            # Save graphs
            convergence_path = f'{results_dir}/graphs/amd_convergence_{exp_id}.png'
            strategies_path = f'{results_dir}/graphs/agent_strategies_{exp_id}.png'
            pg_network_path = f'{results_dir}/graphs/pg_networks_{exp_id}.png'
            
            plot_amd_convergence(metrics, n_agents, save_path=convergence_path)
            agent_type_used = metrics.get('agent_type', "ppo")
            plot_agent_strategies(agents, agent_type=agent_type_used, n_agents=n_agents,
                                 save_path=strategies_path)
            
            # Plot P & G network outputs
            if 'pg_outputs' in metrics and len(metrics['pg_outputs']) > 0:
                plot_pg_network_outputs(metrics['pg_outputs'], n_agents, save_path=pg_network_path)
                # Print summary at end of experiment
                if exp_idx == total_experiments:
                    print_pg_summary(metrics['pg_outputs'], n_agents)
            
            result['graph_paths'] = {
                'convergence': convergence_path,
                'strategies': strategies_path,
                'pg_networks': pg_network_path if 'pg_outputs' in metrics else None
            }
            
            # Save detailed log
            log_path = f'{results_dir}/logs/{exp_id}.json'
            with open(log_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            all_results.append(result)
            
            print(f"\n✓ Experiment {exp_idx} completed")
            print(f"  Final theta: {metrics['final_avg_theta']:.3f}")
            print(f"  Final efficiency: {metrics['final_efficiency']:.3f}")
            print(f"  Final revenue: {metrics['final_revenue']:.3f}")
            print(f"  Graphs saved to: {results_dir}/graphs/")
            
        except Exception as e:
            print(f"\n✗ Experiment {exp_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total_experiments,
        'completed': len(all_results),
        'best_result': best_result,
        'all_results': [
            {
                'exp_id': r['exp_id'],
                'parameters': r['parameters'],
                'results': r['results']
            }
            for r in all_results
        ]
    }
    
    summary_path = f'{results_dir}/experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SWEEP COMPLETE")
    print("=" * 80)
    print(f"Completed: {len(all_results)}/{total_experiments}")
    print("\nBest Result (by revenue):")
    if best_result:
        print(f"  Experiment ID: {best_result['exp_id']}")
        print(f"  Parameters: {best_result['parameters']}")
        print(f"  Final Revenue: {best_result['results']['final_revenue']:.4f}")
        print(f"  Final Theta: {best_result['results']['final_avg_theta']:.4f}")
        print(f"  Final Efficiency: {best_result['results']['final_efficiency']:.4f}")
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"  Graphs: {results_dir}/graphs/")
    print(f"  Logs: {results_dir}/logs/")
    
    return all_results, best_result


if __name__ == "__main__":
    
    results, best = run_experiment_sweep(
        n_agents=10,
        n_rounds=30000,
        learning_rates=[5e-3, 5e-4],
        update_epochs_list=[4],
        gammas=[0.10, 0.99],
        info_types=[InformationType.FULL_REVELATION, InformationType.MINIMAL],
        training_interval=[200, 1000]  
    )
   