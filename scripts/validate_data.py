#!/usr/bin/env python
"""Validate datasets, write quality reports, profiles and EDA figures.

Usage::

    python scripts/validate_data.py
    python scripts/validate_data.py --task heart_disease --no-figures
"""

from __future__ import annotations

import argparse
import sys

from disease_prediction.config import TASKS, get_settings
from disease_prediction.data.eda import generate_eda_figures
from disease_prediction.data.loaders import load_dataset
from disease_prediction.data.quality import profile_frame, save_profile, write_quality_report
from disease_prediction.data.registry import get_schema
from disease_prediction.data.validate import validate_dataframe
from disease_prediction.logging_config import configure_logging, get_logger

logger = get_logger("scripts.validate_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--no-figures", action="store_true", help="Skip EDA figure generation.")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    tasks = list(TASKS) if args.task == "all" else [args.task]

    failed = False
    for task in tasks:
        schema = get_schema(task)
        df = load_dataset(task, settings)
        report = validate_dataframe(df, schema)
        profile = profile_frame(df, schema)

        profile_path = save_profile(task, df, schema, settings)
        report_path = write_quality_report(
            report, settings.metrics_dir / f"{task}_quality.md", profile=profile
        )
        print(report.format_text())
        print(f"  profile -> {profile_path}")
        print(f"  report  -> {report_path}\n")

        if not args.no_figures:
            for figure in generate_eda_figures(task, df, schema, settings.figures_dir):
                print(f"  figure  -> {figure}")

        failed = failed or not report.ok

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
