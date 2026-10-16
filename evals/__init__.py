"""
Evaluation package.

The public helpers are re-exported lazily. Importing the package, or a light
submodule such as evals.retrieval_metrics, must not load ragas, torch or the
API client; only touching one of these names does.
"""

import importlib

_LAZY = {
    "run_pipeline": "evals.pipeline",
    "load_golden_dataset": "evals.pipeline",
    "run_guardrails_eval": "evals.guardrails_eval",
    "compute_guardrails_metrics": "evals.guardrails_eval",
    "run_all_metrics": "evals.metrics",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'evals' has no attribute {name!r}")
