#!/usr/bin/env python
"""Train models for one or more tasks and persist artifacts.

Usage::

    python scripts/train.py                                   # all tasks, all models
    python scripts/train.py --task heart_disease --n-iter 20
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --task diabetes --models logistic_regression xgboost

Values come from (lowest to highest precedence): built-in defaults,
``configs/train.yaml`` (or ``--config``), then explicit CLI flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from disease_prediction.config import TASKS, get_settings
from disease_prediction.evaluation.report import write_comparison_table, write_metrics_json
from disease_prediction.logging_config import configure_logging, get_logger
from disease_prediction.models.registry import model_spec
from disease_prediction.models.train import TrainingConfig, train_task

logger = get_logger("scripts.train")


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(cli: Any, file_value: Any, default: Any) -> Any:
    """CLI value wins, then config file, then built-in default."""
    if cli is not None:
        return cli
    if file_value is not None:
        return file_value
    return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names.")
    parser.add_argument(
        "--n-iter", type=int, default=None, help="RandomizedSearchCV budget per model."
    )
    parser.add_argument("--cv", type=int, default=None, help="CV folds for model selection.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--threshold-objective",
        default=None,
        choices=["f1", "youden", "sensitivity_at_specificity"],
    )
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Do not persist artifacts.")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()

    file_cfg = _load_config(
        args.config if args.config is not None else settings.config_dir / "train.yaml"
    )
    per_task: dict[str, dict[str, Any]] = file_cfg.get("per_task", {}) or {}

    seed = _resolve(args.seed, file_cfg.get("seed"), settings.random_seed)
    calibration = None if args.no_calibration else file_cfg.get("calibration_method", "sigmoid")

    models_arg = args.models if args.models is not None else file_cfg.get("models")
    if models_arg:
        for name in models_arg:  # fail fast on a typo
            model_spec(name)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for task in tasks:
        overrides = per_task.get(task, {}) or {}
        config = TrainingConfig(
            task=task,
            models=_resolve(models_arg, overrides.get("models"), None),
            n_iter=_resolve(args.n_iter, overrides.get("n_iter", file_cfg.get("n_iter")), 12),
            cv_folds=_resolve(args.cv, overrides.get("cv", file_cfg.get("cv")), 5),
            seed=int(seed),
            threshold_objective=_resolve(
                args.threshold_objective,
                overrides.get("threshold_objective", file_cfg.get("threshold_objective")),
                "youden",
            ),
            calibration_method=calibration,
            save=not args.no_save,
        )
        print(f"\n=== Training {task} ===")
        result = train_task(config, settings)
        for row in result.comparison_table():
            print(
                f"  {row['display_name']:<28} val_roc_auc={row['val_roc_auc']:.4f} "
                f"val_f1={row['val_f1']:.4f} thr={row['threshold']:.3f} cal={row['calibrated']}"
            )
        print(f"  champion: {result.champion}")
        print(f"  rationale: {result.rationale}")
        if not args.no_save:
            print(f"  table -> {write_comparison_table(result, settings)}")
            print(f"  json  -> {write_metrics_json(result, settings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
