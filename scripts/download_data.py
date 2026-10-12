#!/usr/bin/env python
"""Download the public datasets and record provenance.

Usage::

    python scripts/download_data.py            # all tasks
    python scripts/download_data.py --task diabetes --force
"""

from __future__ import annotations

import argparse
import sys

from disease_prediction.config import TASKS, get_settings
from disease_prediction.data.download import download_dataset, ensure_heart_vendored
from disease_prediction.logging_config import configure_logging, get_logger

logger = get_logger("scripts.download_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    tasks = list(TASKS) if args.task == "all" else [args.task]

    for task in tasks:
        if task == "heart_disease":
            path = ensure_heart_vendored(settings)
            print(f"[heart_disease] vendored/ready at {path}")
        elif task == "diabetes":
            written = download_dataset("diabetes", settings, force=args.force)
            print(f"[diabetes] downloaded {[str(p) for p in written]}")
        elif task == "breast_cancer":
            print("[breast_cancer] bundled with scikit-learn — no download required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
