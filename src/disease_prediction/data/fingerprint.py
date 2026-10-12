"""Dataset fingerprinting for artifact provenance.

A fingerprint is a deterministic SHA-256 over the *canonical* dataframe (sorted
columns, fixed float formatting) rather than the raw bytes, so it stays stable
across minor file formatting differences and also works for datasets that are
loaded from a library instead of disk (e.g. WDBC via scikit-learn).
"""

from __future__ import annotations

import hashlib

import pandas as pd

from disease_prediction.config import Settings
from disease_prediction.data.loaders import load_dataset


def canonical_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 over a canonical CSV rendering of *df*."""
    ordered = df.reindex(sorted(df.columns), axis=1)
    payload = ordered.to_csv(index=False, float_format="%.6f").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(task: str, settings: Settings | None = None) -> str:
    """Fingerprint the canonical dataset for *task*."""
    return canonical_fingerprint(load_dataset(task, settings))
