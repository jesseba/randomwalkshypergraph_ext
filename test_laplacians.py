from src.data_loading import load_halo_data
from src.evaluations import evaluate_head_to_head
from src.visualization import plot_results
from src.laplacian_constructions import (
    RandomWalkLaplacian, 
    ZhouLaplacian,
    ChanLaplacian
)
import os
import json
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt


def setup_output_directories():
    """Create directories for storing results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create main results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create timestamped directory for this run
    run_dir = os.path.join('results', timestamp)
    os.makedirs(run_dir)
    
    # Create subdirectories
    plots_dir = os.path.join(run_dir, 'plots')
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(plots_dir)
    os.makedirs(metrics_dir)
    
    return run_dir, plots_dir, metrics_dir

def save_results(results: dict, run_dir: str):
    """Save results to files"""
    # Save raw results as JSON
    results_file = os.path.join(run_dir, 'metrics', 'results.json')
    with open(results_file, 'w') as f:
        # Convert numpy values to float for JSON serialization
        serializable_results = {
            k: {
                'accuracy': float(v['accuracy']),
                'total_matches': int(v['total_matches'])
            }
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=4)
    
    # Save summary as text
    summary_file = os.path.join(run_dir, 'metrics', 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Laplacian Comparison Results\n")
        f.write("===========================\n\n")
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.3f}\n")
            f.write(f"  Total Matches: {result['total_matches']}\n\n")

# Modify visualization.py:
def plot_results(results: Dict, plots_dir: str):
    """Plot and save comparison results"""
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(names)), accuracies)
    plt.xticks(range(len(names)), names, rotation=45)
    plt.ylabel('Prediction Accuracy')
    plt.title('Comparison of Laplacian Constructions')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'accuracy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create accuracy vs beta plot if Chan results exist
    chan_results = {k: v for k, v in results.items() if 'chan_beta' in k}
    if chan_results:
        plt.figure(figsize=(10, 6))
        betas = [float(k.split('_')[-1]) for k in chan_results.keys()]
        accs = [v['accuracy'] for v in chan_results.values()]
        
        plt.plot(betas, accs, 'o-')
        plt.xlabel('Beta Value')
        plt.ylabel('Accuracy')
        plt.title('Effect of Beta Parameter on Chan Laplacian Performance')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        beta_plot_path = os.path.join(plots_dir, 'beta_effect.png')
        plt.savefig(beta_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def compare_laplacians(beta_values: List[float] = [0.3, 0.5, 0.7]):
    """Compare different Laplacian constructions"""
    universe, matches = load_halo_data()
    
    results = {}
    laplacians = {
        'random_walk': RandomWalkLaplacian(universe, matches),
        'zhou': ZhouLaplacian(universe, matches)
    }
    
    for beta in beta_values:
        laplacians[f'chan_beta_{beta}'] = ChanLaplacian(universe, matches, beta)
    
    for name, lap in laplacians.items():
        print(f"Evaluating {name}...")
        L = lap.compute_laplacian()
        P = lap.compute_transition_matrix(L)
        rankings = lap.compute_pagerank(P, r=0.4)
        results[name] = evaluate_head_to_head(rankings, universe)
        
    return results

if __name__ == '__main__':
    # Setup output directories
    run_dir, plots_dir, metrics_dir = setup_output_directories()
    
    # Run comparisons
    results = compare_laplacians()
    
    # Save results
    save_results(results, run_dir)
    
    # Generate and save plots
    plot_results(results, plots_dir)
    
    print(f"\nResults saved to: {run_dir}")