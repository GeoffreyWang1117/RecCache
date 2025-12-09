"""Visualization utilities for RecCache analysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def check_plotting():
    """Check if plotting is available."""
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn are required for plotting. "
                         "Install with: pip install matplotlib seaborn")


@dataclass
class ExperimentResults:
    """Container for experiment results."""

    name: str
    hit_rates: List[float]
    ndcg_scores: List[float]
    latencies: List[float]
    parameters: List[float]  # e.g., n_clusters, cache_size
    parameter_name: str


def plot_cache_tradeoff(
    hit_rates: List[float],
    quality_losses: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Cache Hit Rate vs Quality Loss Tradeoff",
    save_path: Optional[str] = None,
):
    """
    Plot the tradeoff between cache hit rate and quality loss.

    Args:
        hit_rates: List of cache hit rates
        quality_losses: List of quality losses (e.g., NDCG degradation)
        labels: Optional labels for each point
        title: Plot title
        save_path: Path to save the figure
    """
    check_plotting()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    scatter = ax.scatter(hit_rates, quality_losses, s=100, c=range(len(hit_rates)),
                        cmap='viridis', alpha=0.7, edgecolors='black')

    # Add labels
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (hit_rates[i], quality_losses[i]),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=9)

    # Pareto frontier
    pareto_x, pareto_y = _compute_pareto_frontier(hit_rates, quality_losses)
    ax.plot(pareto_x, pareto_y, 'r--', alpha=0.5, label='Pareto Frontier')

    ax.set_xlabel('Cache Hit Rate', fontsize=12)
    ax.set_ylabel('Quality Loss (NDCG Degradation)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ideal point annotation
    ax.annotate('Ideal: High hit rate,\nLow quality loss',
               xy=(0.9, 0.02), fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_latency_distribution(
    cache_hit_latencies: List[float],
    cache_miss_latencies: List[float],
    fresh_latencies: Optional[List[float]] = None,
    title: str = "Latency Distribution",
    save_path: Optional[str] = None,
):
    """
    Plot latency distributions for cache hits, misses, and fresh computation.
    """
    check_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    labels = []

    if cache_hit_latencies:
        data.append(cache_hit_latencies)
        labels.append(f'Cache Hit\n(median: {np.median(cache_hit_latencies):.2f}ms)')

    if cache_miss_latencies:
        data.append(cache_miss_latencies)
        labels.append(f'Cache Miss\n(median: {np.median(cache_miss_latencies):.2f}ms)')

    if fresh_latencies:
        data.append(fresh_latencies)
        labels.append(f'Fresh Compute\n(median: {np.median(fresh_latencies):.2f}ms)')

    # Box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color the boxes
    colors = ['#90EE90', '#FFB6C1', '#87CEEB']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add speedup annotation if applicable
    if cache_hit_latencies and fresh_latencies:
        speedup = np.median(fresh_latencies) / np.median(cache_hit_latencies)
        ax.annotate(f'Speedup: {speedup:.1f}x',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=12, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_cluster_analysis(
    cluster_sizes: List[int],
    cluster_qualities: List[float],
    title: str = "Cluster Analysis",
    save_path: Optional[str] = None,
):
    """
    Plot cluster size distribution and quality correlation.
    """
    check_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cluster size distribution
    ax1 = axes[0]
    ax1.hist(cluster_sizes, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(cluster_sizes), color='red', linestyle='--',
                label=f'Mean: {np.mean(cluster_sizes):.1f}')
    ax1.set_xlabel('Cluster Size', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Cluster Size Distribution', fontsize=12)
    ax1.legend()

    # Right: Size vs Quality scatter
    ax2 = axes[1]
    ax2.scatter(cluster_sizes, cluster_qualities, alpha=0.6, edgecolors='black')

    # Trend line
    z = np.polyfit(cluster_sizes, cluster_qualities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(cluster_sizes), max(cluster_sizes), 100)
    ax2.plot(x_line, p(x_line), 'r--', alpha=0.7, label='Trend')

    ax2.set_xlabel('Cluster Size', fontsize=12)
    ax2.set_ylabel('Avg NDCG', fontsize=12)
    ax2.set_title('Cluster Size vs Quality', fontsize=12)
    ax2.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_parameter_sensitivity(
    results: ExperimentResults,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot how metrics change with a parameter (e.g., n_clusters, cache_size).
    """
    check_plotting()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = results.parameters

    # Hit Rate
    ax1 = axes[0]
    ax1.plot(params, results.hit_rates, 'o-', color='green', linewidth=2, markersize=8)
    ax1.set_xlabel(results.parameter_name, fontsize=11)
    ax1.set_ylabel('Hit Rate', fontsize=11)
    ax1.set_title('Cache Hit Rate', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # NDCG
    ax2 = axes[1]
    ax2.plot(params, results.ndcg_scores, 'o-', color='blue', linewidth=2, markersize=8)
    ax2.set_xlabel(results.parameter_name, fontsize=11)
    ax2.set_ylabel('NDCG@10', fontsize=11)
    ax2.set_title('Recommendation Quality', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Latency
    ax3 = axes[2]
    ax3.plot(params, results.latencies, 'o-', color='orange', linewidth=2, markersize=8)
    ax3.set_xlabel(results.parameter_name, fontsize=11)
    ax3.set_ylabel('Avg Latency (ms)', fontsize=11)
    ax3.set_title('Average Latency', fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title or f"Sensitivity Analysis: {results.name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_traffic_pattern_comparison(
    patterns: List[str],
    hit_rates: List[float],
    latencies_p50: List[float],
    latencies_p95: List[float],
    title: str = "Traffic Pattern Comparison",
    save_path: Optional[str] = None,
):
    """
    Compare cache performance across different traffic patterns.
    """
    check_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(patterns))
    width = 0.35

    # Hit Rate
    ax1 = axes[0]
    bars = ax1.bar(x, hit_rates, width, color='steelblue', edgecolor='black')
    ax1.set_ylabel('Hit Rate', fontsize=11)
    ax1.set_title('Cache Hit Rate by Traffic Pattern', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(patterns)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, hit_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=10)

    # Latencies
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, latencies_p50, width, label='P50', color='lightgreen', edgecolor='black')
    bars2 = ax2.bar(x + width/2, latencies_p95, width, label='P95', color='salmon', edgecolor='black')
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.set_title('Latency by Traffic Pattern', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(patterns)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_time_series_metrics(
    timestamps: List[float],
    hit_rates: List[float],
    latencies: List[float],
    window_size: int = 100,
    title: str = "Metrics Over Time",
    save_path: Optional[str] = None,
):
    """
    Plot metrics over time with rolling average.
    """
    check_plotting()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Convert to relative time
    times = np.array(timestamps) - timestamps[0]

    # Rolling average
    def rolling_avg(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Hit Rate
    ax1 = axes[0]
    ax1.plot(times, hit_rates, alpha=0.3, color='blue', label='Raw')
    if len(hit_rates) > window_size:
        rolling = rolling_avg(hit_rates, window_size)
        ax1.plot(times[window_size-1:], rolling, color='blue', linewidth=2,
                label=f'Rolling Avg ({window_size})')
    ax1.set_ylabel('Hit Rate', fontsize=11)
    ax1.set_title('Cache Hit Rate', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Latency
    ax2 = axes[1]
    ax2.plot(times, latencies, alpha=0.3, color='orange', label='Raw')
    if len(latencies) > window_size:
        rolling = rolling_avg(latencies, window_size)
        ax2.plot(times[window_size-1:], rolling, color='orange', linewidth=2,
                label=f'Rolling Avg ({window_size})')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.set_title('Response Latency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_user_embedding_clusters(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "User Embedding Clusters (t-SNE)",
    save_path: Optional[str] = None,
    max_points: int = 5000,
):
    """
    Visualize user embeddings with cluster coloring using t-SNE.
    """
    check_plotting()

    from sklearn.manifold import TSNE

    # Sample if too many points
    if len(embeddings) > max_points:
        idx = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        cluster_labels = cluster_labels[idx]

    # t-SNE reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=cluster_labels, cmap='tab20', alpha=0.6, s=20)

    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title(title, fontsize=14)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def _compute_pareto_frontier(
    x_values: List[float],
    y_values: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Compute Pareto frontier for minimizing y while maximizing x.
    """
    points = sorted(zip(x_values, y_values), key=lambda p: -p[0])

    pareto_x = []
    pareto_y = []
    min_y = float('inf')

    for x, y in points:
        if y < min_y:
            pareto_x.append(x)
            pareto_y.append(y)
            min_y = y

    return pareto_x, pareto_y


def generate_report(
    results: Dict,
    output_dir: str = "results",
    include_plots: bool = True,
):
    """
    Generate a comprehensive report with all visualizations.

    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save plots and report
        include_plots: Whether to generate plots
    """
    import os
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("# RecCache Experiment Report\n")
    report_lines.append(f"Generated automatically\n\n")

    # Summary metrics
    if "summary" in results:
        report_lines.append("## Summary Metrics\n")
        for key, value in results["summary"].items():
            if isinstance(value, float):
                report_lines.append(f"- **{key}**: {value:.4f}\n")
            else:
                report_lines.append(f"- **{key}**: {value}\n")
        report_lines.append("\n")

    # Generate plots if requested
    if include_plots and PLOTTING_AVAILABLE:
        report_lines.append("## Visualizations\n")

        if "tradeoff" in results:
            plot_cache_tradeoff(
                results["tradeoff"]["hit_rates"],
                results["tradeoff"]["quality_losses"],
                labels=results["tradeoff"].get("labels"),
                save_path=str(output_path / "tradeoff.png"),
            )
            report_lines.append("![Tradeoff](tradeoff.png)\n\n")
            plt.close()

        if "latency" in results:
            plot_latency_distribution(
                results["latency"].get("cache_hit", []),
                results["latency"].get("cache_miss", []),
                results["latency"].get("fresh", []),
                save_path=str(output_path / "latency.png"),
            )
            report_lines.append("![Latency](latency.png)\n\n")
            plt.close()

    # Write report
    report_path = output_path / "report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    print(f"Report saved to {report_path}")
    return report_path
