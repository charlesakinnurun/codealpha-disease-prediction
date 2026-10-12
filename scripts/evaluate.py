#!/usr/bin/env python
"""Evaluate trained champion models on the held-out test split.

Reproduces the exact train/validation/test split used during training (same seed
and sizes), reloads the persisted artifacts, and regenerates test metrics,
figures and a short report. This is the command CI and reviewers can run to
independently confirm the reported numbers.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --task heart_disease --no-figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from disease_prediction.config import TASKS, get_settings
from disease_prediction.data.loaders import load_dataset
from disease_prediction.data.registry import get_schema
from disease_prediction.evaluation.metrics import binary_metrics, bootstrap_ci
from disease_prediction.evaluation.plots import (
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_pr,
    plot_roc,
)
from disease_prediction.explainability.explain import global_explanations
from disease_prediction.logging_config import configure_logging, get_logger
from disease_prediction.models.persistence import load_artifact
from disease_prediction.models.train import TrainingConfig, split_data

logger = get_logger("scripts.evaluate")


def evaluate_task(task: str, *, make_figures: bool, seed: int) -> dict:
    settings = get_settings()
    schema = get_schema(task)
    frame = load_dataset(task, settings)
    config = TrainingConfig(task=task, seed=seed)
    _, _, test = split_data(frame, config)
    feature_names = schema.feature_names

    from disease_prediction.models.persistence import get_champion

    champion_name = get_champion(task, settings)
    if champion_name is None:
        raise SystemExit(f"No champion recorded for {task}. Run scripts/train.py first.")

    loaded = load_artifact(task, champion_name, settings)
    x_test = test[feature_names]
    y_test = test["target"].to_numpy()
    proba = loaded.pipeline.predict_proba(x_test)[:, list(loaded.pipeline.classes_).index(1)]
    metrics = binary_metrics(y_test, proba, threshold=loaded.threshold)
    metrics["roc_auc_ci95"] = list(bootstrap_ci(y_test, proba, "roc_auc", seed=seed))

    payload = {
        "task": task,
        "champion": champion_name,
        "model_version": loaded.model_version,
        "threshold": loaded.threshold,
        "n_test": int(len(y_test)),
        "metrics": metrics,
    }

    figures: list[Path] = []
    if make_figures:
        fig_dir = settings.figures_dir / task
        figures.append(
            plot_confusion(
                metrics, fig_dir / "champion_confusion.png", f"{task}: champion confusion"
            )
        )
        figures.append(plot_roc(y_test, proba, fig_dir / "champion_roc.png", f"{task}: ROC"))
        figures.append(plot_pr(y_test, proba, fig_dir / "champion_pr.png", f"{task}: PR"))
        figures.append(
            plot_calibration(
                y_test, proba, fig_dir / "champion_calibration.png", f"{task}: calibration"
            )
        )
        try:
            explanations = global_explanations(
                loaded.pipeline, x_test, y_test, seed=seed, background=x_test
            )
            perm = explanations["permutation"].head(15)
            figures.append(
                plot_importance(
                    list(perm["feature"]),
                    [float(v) for v in perm["importance_mean"]],
                    fig_dir / "champion_permutation_importance.png",
                    f"{task}: permutation importance (test split)",
                )
            )
            if "shap" in explanations:
                top = explanations["shap"].head(15)
                figures.append(
                    plot_importance(
                        list(top["feature"]),
                        [float(v) for v in top["mean_abs_shap"]],
                        fig_dir / "champion_shap_importance.png",
                        f"{task}: mean |SHAP|",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - explanations are best-effort
            logger.warning("explanation figures failed task=%s err=%s", task, exc)

    out = settings.metrics_dir / f"{task}_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["_paths"] = [str(p) for p in figures]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    seed = args.seed if args.seed is not None else get_settings().random_seed

    for task in tasks:
        payload = evaluate_task(task, make_figures=not args.no_figures, seed=seed)
        m = payload["metrics"]
        print(f"\n=== {task} (champion: {payload['champion']}) ===")
        print(
            f"  n_test={payload['n_test']}  roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}  "
            f"f1={m['f1']:.4f}  sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}  "
            f"brier={m['brier']:.4f}"
        )
        print(f"  confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
        for path in payload["_paths"]:
            print(f"  figure -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
