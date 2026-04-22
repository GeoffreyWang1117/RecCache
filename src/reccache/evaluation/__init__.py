"""Evaluation module for measuring cache and recommendation quality."""

from reccache.evaluation.metrics import RecommendationMetrics, CacheEvaluator, SpeculativeMetrics
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig
from reccache.evaluation.experiment import (
    ExperimentResult,
    AggregatedResult,
    SignificanceTestResult,
    StatisticalTester,
    ExperimentRunner,
    AblationStudy,
    ParameterSensitivityAnalysis,
    UserGroupAnalysis,
)

__all__ = [
    "RecommendationMetrics",
    "CacheEvaluator",
    "SpeculativeMetrics",
    "OnlineSimulator",
    "SimulationConfig",
    # Experiment framework
    "ExperimentResult",
    "AggregatedResult",
    "SignificanceTestResult",
    "StatisticalTester",
    "ExperimentRunner",
    "AblationStudy",
    "ParameterSensitivityAnalysis",
    "UserGroupAnalysis",
]
