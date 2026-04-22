"""Experiment framework with statistical testing and ablation studies."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from scipy import stats
from tqdm import tqdm


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    metrics: Dict[str, float]
    run_id: int = 0
    seed: int = 42
    config: Dict = field(default_factory=dict)
    runtime_seconds: float = 0.0


@dataclass
class AggregatedResult:
    """Aggregated result across multiple runs."""

    mean: Dict[str, float]
    std: Dict[str, float]
    all_runs: List[Dict[str, float]]
    n_runs: int
    config: Dict = field(default_factory=dict)

    def __str__(self):
        lines = []
        for metric, value in self.mean.items():
            std = self.std.get(metric, 0.0)
            lines.append(f"  {metric}: {value:.4f} +/- {std:.4f}")
        return "\n".join(lines)


@dataclass
class SignificanceTestResult:
    """Result from statistical significance test."""

    test_name: str
    method_a: str
    method_b: str
    metric: str
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    def __str__(self):
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (f"{self.method_a} vs {self.method_b} ({self.metric}): "
                f"p={self.p_value:.4f}{sig} ({'significant' if self.significant else 'not significant'})")


class StatisticalTester:
    """Statistical significance testing for recommendation experiments."""

    @staticmethod
    def paired_ttest(
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05,
    ) -> Tuple[float, bool]:
        """
        Paired t-test for comparing two methods.

        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alpha: Significance level

        Returns:
            (p-value, is_significant)
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        if len(scores_a) < 2:
            return 1.0, False

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        return p_value, p_value < alpha

    @staticmethod
    def wilcoxon_test(
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05,
    ) -> Tuple[float, bool]:
        """
        Wilcoxon signed-rank test (non-parametric alternative to t-test).

        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alpha: Significance level

        Returns:
            (p-value, is_significant)
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        if len(scores_a) < 2:
            return 1.0, False

        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            # All differences are zero
            return 1.0, False

        return p_value, p_value < alpha

    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values
            alpha: Original significance level

        Returns:
            List of booleans indicating significance after correction
        """
        corrected_alpha = alpha / len(p_values)
        return [p < corrected_alpha for p in p_values]

    @staticmethod
    def cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
        """
        Compute Cohen's d effect size.

        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B

        Returns:
            Cohen's d (effect size)
        """
        diff = np.array(scores_a) - np.array(scores_b)
        return np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

    @staticmethod
    def confidence_interval(
        scores: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean.

        Args:
            scores: List of scores
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            (lower_bound, upper_bound)
        """
        n = len(scores)
        mean = np.mean(scores)
        se = stats.sem(scores)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h

    def compare_methods(
        self,
        results: Dict[str, List[Dict[str, float]]],
        baseline: str,
        metric: str,
        alpha: float = 0.05,
        test: str = "ttest",
    ) -> List[SignificanceTestResult]:
        """
        Compare all methods against a baseline.

        Args:
            results: Dict mapping method name to list of run results
            baseline: Name of baseline method
            metric: Metric to compare
            alpha: Significance level
            test: Test type ("ttest" or "wilcoxon")

        Returns:
            List of significance test results
        """
        if baseline not in results:
            raise ValueError(f"Baseline '{baseline}' not found")

        baseline_scores = [r[metric] for r in results[baseline]]
        test_results = []

        for method, runs in results.items():
            if method == baseline:
                continue

            method_scores = [r[metric] for r in runs]

            if test == "ttest":
                p_value, significant = self.paired_ttest(baseline_scores, method_scores, alpha)
            else:
                p_value, significant = self.wilcoxon_test(baseline_scores, method_scores, alpha)

            effect_size = self.cohens_d(method_scores, baseline_scores)

            test_results.append(SignificanceTestResult(
                test_name=test,
                method_a=method,
                method_b=baseline,
                metric=metric,
                p_value=p_value,
                significant=significant,
                alpha=alpha,
                effect_size=effect_size,
            ))

        return test_results


class ExperimentRunner:
    """Run experiments with multiple seeds and aggregate results."""

    def __init__(
        self,
        n_runs: int = 5,
        seeds: Optional[List[int]] = None,
        output_dir: str = "results",
    ):
        self.n_runs = n_runs
        self.seeds = seeds or list(range(42, 42 + n_runs))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tester = StatisticalTester()

    def run_experiment(
        self,
        experiment_fn: Callable[[int], Dict[str, float]],
        name: str = "experiment",
        verbose: bool = True,
    ) -> AggregatedResult:
        """
        Run experiment multiple times with different seeds.

        Args:
            experiment_fn: Function that takes seed and returns metrics dict
            name: Experiment name
            verbose: Print progress

        Returns:
            Aggregated results
        """
        all_runs = []

        iterator = tqdm(self.seeds, desc=name) if verbose else self.seeds

        for seed in iterator:
            start_time = time.time()
            metrics = experiment_fn(seed)
            runtime = time.time() - start_time

            all_runs.append({
                **metrics,
                "_seed": seed,
                "_runtime": runtime,
            })

        # Aggregate
        metric_names = [k for k in all_runs[0].keys() if not k.startswith("_")]

        mean = {m: np.mean([r[m] for r in all_runs]) for m in metric_names}
        std = {m: np.std([r[m] for r in all_runs]) for m in metric_names}

        return AggregatedResult(
            mean=mean,
            std=std,
            all_runs=all_runs,
            n_runs=len(all_runs),
        )

    def run_comparison(
        self,
        methods: Dict[str, Callable[[int], Dict[str, float]]],
        baseline: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, AggregatedResult], List[SignificanceTestResult]]:
        """
        Run comparison experiment across multiple methods.

        Args:
            methods: Dict mapping method name to experiment function
            baseline: Baseline method for significance testing
            verbose: Print progress

        Returns:
            (aggregated_results, significance_tests)
        """
        results = {}

        for name, fn in methods.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Running: {name}")
                print('='*50)

            results[name] = self.run_experiment(fn, name, verbose)

        # Significance testing
        sig_tests = []
        if baseline and len(methods) > 1:
            all_runs = {name: res.all_runs for name, res in results.items()}

            # Test each metric
            metrics = [k for k in results[baseline].mean.keys() if not k.startswith("_")]

            for metric in metrics:
                tests = self.tester.compare_methods(all_runs, baseline, metric)
                sig_tests.extend(tests)

        return results, sig_tests

    def save_results(
        self,
        results: Dict[str, AggregatedResult],
        sig_tests: List[SignificanceTestResult],
        filename: str = "experiment_results.json",
    ):
        """Save results to JSON file."""
        output = {
            "results": {
                name: {
                    "mean": res.mean,
                    "std": res.std,
                    "n_runs": res.n_runs,
                    "all_runs": res.all_runs,
                }
                for name, res in results.items()
            },
            "significance_tests": [
                {
                    "method_a": t.method_a,
                    "method_b": t.method_b,
                    "metric": t.metric,
                    "p_value": t.p_value,
                    "significant": t.significant,
                    "effect_size": t.effect_size,
                }
                for t in sig_tests
            ],
        }

        with open(self.output_dir / filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

    def print_comparison_table(
        self,
        results: Dict[str, AggregatedResult],
        metrics: List[str],
        sig_tests: Optional[List[SignificanceTestResult]] = None,
    ):
        """Print a comparison table."""
        # Build significance lookup
        sig_lookup = {}
        if sig_tests:
            for t in sig_tests:
                key = (t.method_a, t.metric)
                sig_lookup[key] = t.p_value

        # Header
        header = ["Method"] + metrics
        print("\n" + "="*80)
        print(f"{'Method':<20}", end="")
        for m in metrics:
            print(f"{m:>15}", end="")
        print()
        print("-"*80)

        # Rows
        for name, res in results.items():
            print(f"{name:<20}", end="")
            for m in metrics:
                mean_val = res.mean.get(m, 0)
                std_val = res.std.get(m, 0)

                # Add significance marker
                sig_marker = ""
                if (name, m) in sig_lookup:
                    p = sig_lookup[(name, m)]
                    if p < 0.001:
                        sig_marker = "***"
                    elif p < 0.01:
                        sig_marker = "**"
                    elif p < 0.05:
                        sig_marker = "*"

                print(f"{mean_val:.4f}+/-{std_val:.4f}{sig_marker:>3}", end="")
            print()

        print("="*80)
        print("Significance: * p<0.05, ** p<0.01, *** p<0.001")


class AblationStudy:
    """Framework for ablation studies."""

    def __init__(
        self,
        base_experiment: Callable[[Dict], Dict[str, float]],
        base_config: Dict,
        n_runs: int = 5,
    ):
        """
        Args:
            base_experiment: Function that takes config dict and returns metrics
            base_config: Full model configuration
            n_runs: Number of runs per variant
        """
        self.base_experiment = base_experiment
        self.base_config = base_config.copy()
        self.n_runs = n_runs
        self.runner = ExperimentRunner(n_runs=n_runs)

    def run_ablation(
        self,
        ablations: Dict[str, Dict],
        verbose: bool = True,
    ) -> Dict[str, AggregatedResult]:
        """
        Run ablation study.

        Args:
            ablations: Dict mapping ablation name to config changes
                      e.g., {"w/o quality_predictor": {"use_quality_predictor": False}}
            verbose: Print progress

        Returns:
            Dict mapping variant name to aggregated results
        """
        results = {}

        # Run full model
        def full_experiment(seed):
            config = {**self.base_config, "seed": seed}
            return self.base_experiment(config)

        if verbose:
            print("Running full model...")
        results["Full Model"] = self.runner.run_experiment(full_experiment, "Full Model", verbose)

        # Run each ablation
        for ablation_name, config_changes in ablations.items():
            def ablation_experiment(seed, changes=config_changes):
                config = {**self.base_config, **changes, "seed": seed}
                return self.base_experiment(config)

            if verbose:
                print(f"\nRunning {ablation_name}...")
            results[ablation_name] = self.runner.run_experiment(
                ablation_experiment, ablation_name, verbose
            )

        return results

    def print_ablation_table(
        self,
        results: Dict[str, AggregatedResult],
        metrics: List[str],
    ):
        """Print ablation study results."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)

        full_model = results.get("Full Model")
        if not full_model:
            print("Full model results not found")
            return

        print(f"\n{'Variant':<35}", end="")
        for m in metrics:
            print(f"{m:>12}", end="")
        print(f"{'Change':>12}")
        print("-"*80)

        for name, res in results.items():
            print(f"{name:<35}", end="")
            for m in metrics:
                val = res.mean.get(m, 0)
                print(f"{val:.4f}", end="      ")

            # Show change from full model
            if name != "Full Model":
                main_metric = metrics[0]
                full_val = full_model.mean.get(main_metric, 0)
                ablation_val = res.mean.get(main_metric, 0)
                change = ((ablation_val - full_val) / full_val) * 100 if full_val != 0 else 0
                print(f"{change:+.2f}%", end="")
            print()

        print("="*80)


class ParameterSensitivityAnalysis:
    """Analyze sensitivity to hyperparameters."""

    def __init__(
        self,
        experiment_fn: Callable[[Dict], Dict[str, float]],
        base_config: Dict,
        n_runs: int = 3,
    ):
        self.experiment_fn = experiment_fn
        self.base_config = base_config
        self.n_runs = n_runs
        self.runner = ExperimentRunner(n_runs=n_runs)

    def analyze_parameter(
        self,
        param_name: str,
        param_values: List,
        metric: str = "ndcg@10",
        verbose: bool = True,
    ) -> Dict[Any, AggregatedResult]:
        """
        Analyze sensitivity to a single parameter.

        Args:
            param_name: Parameter name in config
            param_values: Values to test
            metric: Metric to focus on
            verbose: Print progress

        Returns:
            Dict mapping param value to results
        """
        results = {}

        for value in param_values:
            def param_experiment(seed, val=value):
                config = {**self.base_config, param_name: val, "seed": seed}
                return self.experiment_fn(config)

            name = f"{param_name}={value}"
            if verbose:
                print(f"\nTesting {name}...")

            results[value] = self.runner.run_experiment(param_experiment, name, verbose)

        return results

    def analyze_multiple_parameters(
        self,
        parameters: Dict[str, List],
        metric: str = "ndcg@10",
        verbose: bool = True,
    ) -> Dict[str, Dict[Any, AggregatedResult]]:
        """
        Analyze sensitivity to multiple parameters.

        Args:
            parameters: Dict mapping param name to list of values
            metric: Metric to focus on
            verbose: Print progress

        Returns:
            Dict mapping param name to results dict
        """
        all_results = {}

        for param_name, values in parameters.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Analyzing: {param_name}")
                print('='*50)

            all_results[param_name] = self.analyze_parameter(
                param_name, values, metric, verbose
            )

        return all_results

    def print_sensitivity_table(
        self,
        param_results: Dict[Any, AggregatedResult],
        param_name: str,
        metrics: List[str],
    ):
        """Print parameter sensitivity table."""
        print(f"\n{'='*60}")
        print(f"Parameter Sensitivity: {param_name}")
        print('='*60)

        print(f"\n{param_name:<15}", end="")
        for m in metrics:
            print(f"{m:>15}", end="")
        print()
        print("-"*60)

        for value, res in sorted(param_results.items(), key=lambda x: x[0]):
            print(f"{str(value):<15}", end="")
            for m in metrics:
                val = res.mean.get(m, 0)
                std = res.std.get(m, 0)
                print(f"{val:.4f}+/-{std:.4f}", end=" ")
            print()

        print("="*60)


class UserGroupAnalysis:
    """Analyze performance across different user groups."""

    @staticmethod
    def group_by_activity(
        user_interactions: Dict[int, int],
        thresholds: List[int] = [10, 50, 200],
    ) -> Dict[str, List[int]]:
        """
        Group users by activity level.

        Args:
            user_interactions: Dict mapping user_id to interaction count
            thresholds: Activity thresholds

        Returns:
            Dict mapping group name to list of user IDs
        """
        groups = {
            f"cold (<{thresholds[0]})": [],
            f"sparse ({thresholds[0]}-{thresholds[1]})": [],
            f"normal ({thresholds[1]}-{thresholds[2]})": [],
            f"active (>{thresholds[2]})": [],
        }

        for user_id, count in user_interactions.items():
            if count < thresholds[0]:
                groups[f"cold (<{thresholds[0]})"].append(user_id)
            elif count < thresholds[1]:
                groups[f"sparse ({thresholds[0]}-{thresholds[1]})"].append(user_id)
            elif count < thresholds[2]:
                groups[f"normal ({thresholds[1]}-{thresholds[2]})"].append(user_id)
            else:
                groups[f"active (>{thresholds[2]})"].append(user_id)

        return groups

    @staticmethod
    def evaluate_per_group(
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, set],
        user_groups: Dict[str, List[int]],
        k: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate recommendations for each user group.

        Args:
            recommendations: User recommendations
            ground_truth: User ground truth
            user_groups: User groups
            k: Cutoff

        Returns:
            Dict mapping group name to metrics
        """
        from reccache.evaluation.metrics import RecommendationMetrics

        group_results = {}

        for group_name, user_ids in user_groups.items():
            group_recs = {u: recommendations[u] for u in user_ids if u in recommendations}
            group_gt = {u: ground_truth[u] for u in user_ids if u in ground_truth}

            if group_recs and group_gt:
                metrics = RecommendationMetrics.evaluate_recommendations(
                    group_recs, group_gt, k=k
                )
                metrics["n_users"] = len(group_recs)
                group_results[group_name] = metrics

        return group_results

    @staticmethod
    def print_group_analysis(group_results: Dict[str, Dict[str, float]], metrics: List[str]):
        """Print user group analysis."""
        print("\n" + "="*80)
        print("USER GROUP ANALYSIS")
        print("="*80)

        print(f"\n{'Group':<30}{'N Users':>10}", end="")
        for m in metrics:
            print(f"{m:>12}", end="")
        print()
        print("-"*80)

        for group_name, results in group_results.items():
            print(f"{group_name:<30}{results.get('n_users', 0):>10}", end="")
            for m in metrics:
                val = results.get(m, 0)
                print(f"{val:.4f}", end="      ")
            print()

        print("="*80)
