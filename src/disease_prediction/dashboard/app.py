"""Streamlit dashboard for the Disease Prediction system.

Run with::

    streamlit run src/disease_prediction/dashboard/app.py

The UI only reads from the trained artifact store; it never trains models and
never persists user input.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from disease_prediction.config import TASKS
from disease_prediction.dashboard.components import (
    DISCLAIMER,
    EXAMPLE_NOTICE,
    champion_metadata,
    champion_metrics,
    confusion_counts,
    contributions_frame,
    metrics_table,
    prediction_label,
    probability_frame,
    run_prediction,
    task_label,
)
from disease_prediction.data.registry import get_schema
from disease_prediction.data.schemas import FeatureKind

st.set_page_config(page_title="Disease Prediction (Research)", page_icon="🧪", layout="wide")


@st.cache_resource
def get_service():
    from disease_prediction.dashboard.components import model_service

    return model_service()


def _render_sidebar() -> str:
    st.sidebar.title("Disease Prediction")
    st.sidebar.caption("Explainable ML · educational & research use only")
    task = st.sidebar.radio(
        "Prediction task",
        options=list(TASKS),
        format_func=task_label,
    )
    st.sidebar.markdown("---")
    st.sidebar.warning(DISCLAIMER)
    return task


def _feature_inputs(task: str) -> dict[str, Any]:
    """Render validated inputs for a task and return the collected values."""
    schema = get_schema(task)
    values: dict[str, Any] = {}

    col_left, col_right = st.columns(2)
    for index, spec in enumerate(schema.features):
        target = col_left if index % 2 == 0 else col_right
        with target:
            label = spec.name + (f" ({spec.unit})" if spec.unit else "")
            help_text = spec.description or None
            if spec.kind is FeatureKind.NUMERIC:
                low = spec.minimum if spec.minimum is not None else 0.0
                high = spec.maximum if spec.maximum is not None else None
                if spec.optional:
                    include = st.checkbox(f"Include {label}", value=False)
                    values[spec.name] = (
                        st.number_input(
                            label, min_value=low, max_value=high, value=None, help=help_text
                        )
                        if include
                        else None
                    )
                else:
                    values[spec.name] = st.number_input(
                        label,
                        min_value=low,
                        max_value=high,
                        value=None,
                        help=help_text,
                        key=f"{task}-{spec.name}",
                    )
            else:
                options = list(spec.allowed_values or [])
                if spec.optional:
                    options_full: list[Any] = [None, *options]
                    choice = st.selectbox(
                        label, options=options_full, help=help_text, key=f"{task}-{spec.name}"
                    )
                    values[spec.name] = choice
                else:
                    values[spec.name] = st.selectbox(
                        label, options=options, help=help_text, key=f"{task}-{spec.name}"
                    )
    return values


def render_prediction_tab(task: str) -> None:
    st.subheader(f"{task_label(task)} prediction")
    st.caption(EXAMPLE_NOTICE)

    schema = get_schema(task)
    with st.form(f"form-{task}"):
        st.markdown("#### Patient features")
        with st.expander("Dataset description", expanded=False):
            st.markdown(f"**Source:** {schema.source_url}")
            st.markdown(f"**Licence:** {schema.license}")
            st.markdown(f"**Target:** {schema.target_meaning}. {schema.target_description}")
            if schema.notes:
                st.info(schema.notes)
        values = _feature_inputs(task)
        explain = st.checkbox("Show feature contributions (slower)", value=False)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    missing = [
        spec.name
        for spec in schema.features
        if spec.kind is FeatureKind.NUMERIC and not spec.optional and values.get(spec.name) is None
    ]
    if missing:
        st.error(f"Please provide values for: {', '.join(missing)}")
        return

    outcome, error = run_prediction(get_service(), task, values, explain=explain)
    if error:
        st.error(f"Prediction failed: {error}")
        return
    assert outcome is not None

    st.success(f"Predicted class: **{prediction_label(outcome)}**")
    if outcome.calibrated:
        st.metric("Probability (positive class)", f"{outcome.probability_positive:.3f}")
    else:
        st.metric("Score (positive class, uncalibrated)", f"{outcome.probability_positive:.3f}")
    st.caption(
        f"Model: `{outcome.model_name}` ({outcome.model_version}) · "
        f"decision threshold = {outcome.threshold:.3f} · target = {outcome.target_meaning}"
    )
    for warning in outcome.warnings:
        st.warning(warning)

    st.markdown("#### Class probabilities")
    st.dataframe(
        probability_frame(outcome).style.format({"probability": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    contributions = contributions_frame(outcome)
    if contributions is not None:
        st.markdown("#### Feature contributions (local)")
        st.dataframe(
            contributions[["feature", "contribution"]], use_container_width=True, hide_index=True
        )
        st.caption(
            "Contributions are model attributions on the transformed feature scale. "
            "They are not causal explanations and can be unstable for correlated features."
        )


def render_performance_tab(task: str) -> None:
    st.subheader(f"{task_label(task)} — model performance")
    metrics = champion_metrics(get_service(), task)
    if metrics is None:
        st.info("No trained model found for this task. Run `python scripts/train.py` first.")
        return

    st.markdown("#### Held-out test metrics")
    st.dataframe(metrics_table(metrics), use_container_width=True, hide_index=True)

    st.markdown("#### Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            confusion_counts(metrics),
            index=["actual_negative", "actual_positive"],
            columns=["pred_negative", "pred_positive"],
        ),
        use_container_width=True,
    )
    meta = champion_metadata(get_service(), task) or {}
    st.caption(f"Test ROC-AUC 95% CI: {metrics.get('roc_auc_ci95')}")
    with st.expander("Model card summary"):
        st.json(
            {
                "model": meta.get("display_name"),
                "selected": meta.get("selection_rationale"),
                "calibrated": meta.get("calibrated"),
                "dataset": meta.get("dataset_id"),
                "licence": meta.get("dataset_license"),
                "trained_at": meta.get("trained_at"),
            }
        )


def render_about_tab() -> None:
    st.subheader("Responsible use")
    st.error(DISCLAIMER)
    st.markdown(
        """
        #### What this system is
        - A reproducible reference implementation of an explainable tabular ML
          pipeline over three public benchmark datasets.
        - A demonstration of leakage-safe preprocessing, calibration, threshold
          selection, evaluation and serving.

        #### What it is not
        - Not a medical device and not clinically validated.
        - Not a diagnosis, prognosis, or treatment recommendation.
        - Not evaluated on the population, equipment or workflow of any hospital.

        #### Known limitations
        - Small, old, demographically narrow benchmark datasets.
        - Labels are proxies (e.g. questionnaire-based diabetes risk).
        - Probabilities are calibrated on tiny validation splits and may not
          transfer to other populations.
        - Feature attributions can be unstable when inputs are correlated.

        See `docs/responsible_use.md` and `docs/model_card.md` for details.
        """
    )


def main() -> None:
    task = _render_sidebar()
    st.title("Disease Prediction from Medical Data")
    st.caption("An explainable machine-learning system — educational and research use only.")

    tab_predict, tab_perf, tab_about = st.tabs(
        ["Predict", "Model performance", "About & responsible use"]
    )
    with tab_predict:
        render_prediction_tab(task)
    with tab_perf:
        render_performance_tab(task)
    with tab_about:
        render_about_tab()


if __name__ == "__main__":  # pragma: no cover
    main()
