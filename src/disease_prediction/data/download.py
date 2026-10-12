"""Dataset acquisition with provenance recording.

Downloads are idempotent and never overwrite a verified file unless ``force`` is
set. Each download writes a ``provenance.json`` capturing the URL, retrieval
timestamp, byte size and SHA-256 of the artefact so that the exact input can be
re-identified later.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from disease_prediction.config import Settings, get_settings
from disease_prediction.logging_config import get_logger

logger = get_logger(__name__)

#: Official UCI download endpoints. Both are CC BY 4.0.
_DOWNLOAD_SOURCES: dict[str, str] = {
    "heart_disease": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    "diabetes": (
        "https://archive.ics.uci.edu/static/public/529/"
        "early+stage+diabetes+risk+prediction+dataset.zip"
    ),
}

#: File inside each archive we ultimately want.
_ARCHIVE_MEMBERS: dict[str, list[str]] = {
    "heart_disease": ["processed.cleveland.data"],
    "diabetes": ["diabetes_data_upload.csv"],
}

_TIMEOUT = (10, 120)


@dataclass
class Provenance:
    """Recorded metadata for a downloaded artefact."""

    task: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    retrieved_at: str
    license: str = "CC BY 4.0 (UCI Machine Learning Repository)"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_dataset(
    task: str,
    settings: Settings | None = None,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> list[Path]:
    """Download and extract a dataset archive, returning written file paths.

    Existing extracted files are kept unless ``force`` is true, which makes CI
    and repeated local runs cheap and reproducible.
    """
    if task not in _DOWNLOAD_SOURCES:
        raise KeyError(f"No download source for task {task!r}")
    settings = settings or get_settings()
    settings.ensure_dirs()
    dest_dir = settings.data_dir / "raw" / task
    dest_dir.mkdir(parents=True, exist_ok=True)

    wanted = _ARCHIVE_MEMBERS[task]
    existing = [dest_dir / name for name in wanted]
    if not force and all(p.exists() for p in existing):
        logger.info("dataset present task=%s dir=%s", task, dest_dir)
        return existing

    url = _DOWNLOAD_SOURCES[task]
    http = session or requests.Session()
    logger.info("downloading task=%s url=%s", task, url)
    response = http.get(url, timeout=_TIMEOUT)
    response.raise_for_status()
    payload = response.content

    written: list[Path] = []
    archive = zipfile.ZipFile(io.BytesIO(payload))
    for member in archive.namelist():
        name = Path(member).name
        if name in wanted:
            target = dest_dir / name
            with archive.open(member) as src:
                data = src.read()
            target.write_bytes(data)
            written.append(target)

    if not written:
        raise RuntimeError(
            f"Archive for {task} did not contain any of {wanted}. Contents: {archive.namelist()}"
        )

    provenance = Provenance(
        task=task,
        url=url,
        filename="+".join(p.name for p in written),
        sha256=_sha256(payload),
        size_bytes=len(payload),
        retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    provenance.write(dest_dir / "provenance.json")
    logger.info(
        "downloaded task=%s files=%s sha256=%s",
        task,
        [p.name for p in written],
        provenance.sha256[:12],
    )
    return written


def ensure_heart_vendored(settings: Settings | None = None) -> Path:
    """Return the vendored Cleveland file, downloading if it is absent.

    The repository ships the small Cleveland file so heart-disease training is
    fully offline; if a fresh clone removed it, fall back to UCI.
    """
    settings = settings or get_settings()
    vendored = settings.data_dir / "heart+disease" / "processed.cleveland.data"
    if vendored.exists():
        return vendored
    logger.info("vendored heart data missing; downloading from UCI")
    download_dataset("heart_disease", settings, force=True)
    return vendored
