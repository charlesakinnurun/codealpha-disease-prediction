#!/usr/bin/env python
"""Generate ``docs/data_dictionary.md`` from the dataset registry.

Keeping the dictionary generated means documentation can never drift from the
schemas enforced at training and inference time.

Usage::

    python scripts/generate_docs.py
"""

from __future__ import annotations

import sys

from disease_prediction.config import REPO_ROOT, TASKS
from disease_prediction.data.registry import all_schemas


def render() -> str:
    lines = [
        "# Data Dictionary",
        "",
        "> Generated from `disease_prediction.data.registry` by "
        "`scripts/generate_docs.py`. Do not edit by hand.",
        "",
        "Feature bounds are inclusive plausibility limits used for validation; "
        "they are wider than the observed dataset range on purpose. "
        "`optional` features may contain missing values, which the training "
        "pipeline imputes.",
        "",
    ]
    for task in TASKS:
        schema = all_schemas()[task]
        lines += [
            f"## {schema.display_name}",
            "",
            f"- **Task id:** `{schema.task}`",
            f"- **Source:** {schema.source_url}",
            f"- **Licence:** {schema.license}",
            f"- **Citation:** {schema.citation}",
            f"- **Target column (raw):** `{schema.target_column}`",
            f"- **Target meaning:** {schema.target_meaning}",
            f"- **Target definition:** {schema.target_description}",
            "",
            "| Feature | Kind | Unit | Allowed / Range | Optional | Description |",
            "| --- | --- | --- | --- | :---: | --- |",
        ]
        for spec in schema.features:
            if spec.allowed_values is not None:
                allowed = ", ".join(str(v) for v in spec.allowed_values)
            elif spec.minimum is not None or spec.maximum is not None:
                allowed = f"[{spec.minimum}, {spec.maximum}]"
            else:
                allowed = "—"
            lines.append(
                f"| `{spec.name}` | {spec.kind.value} | {spec.unit or '—'} | {allowed} | "
                f"{'yes' if spec.optional else 'no'} | {spec.description} |"
            )
        if schema.notes:
            lines += ["", f"**Notes:** {schema.notes}"]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "data_dictionary.md"
    path.write_text(render(), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
