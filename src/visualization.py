import matplotlib.pyplot as plt
from typing import Dict

def plot_results(results: Dict):
    """Plot comparison results"""
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(names)), accuracies)
    plt.xticks(range(len(names)), names, rotation=45)
    plt.ylabel('Prediction Accuracy')
    plt.title('Comparison of Laplacian Constructions')
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()