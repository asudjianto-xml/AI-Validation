"""Vendored EvaluationBatch container from Frontier-Discovery.

This is the single data structure the `aiv` adapters return to the
Frontier-Discovery search engine. It is copied here so `aiv` imports without the
frontier_discovery package installed; when that package is present, its own
EvaluationBatch is structurally identical and interoperates by duck typing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """Result of evaluating one candidate on a batch of inputs.

    - `outputs`, `scores` align 1:1 with the input batch. Scores are
      higher-is-better; the optimizer sums them over a minibatch for acceptance
      and means them over the full valset for tracking and the Pareto frontier.
    - `trajectories` align 1:1 with outputs when `capture_traces=True`, else None.
    - `objective_scores` are per-example objective-name -> score maps, or None
      when the run is single-objective.
    """

    outputs: list
    scores: list
    trajectories: list | None = None
    objective_scores: list | None = None
    num_metric_calls: int | None = None
