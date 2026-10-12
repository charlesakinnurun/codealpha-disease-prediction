# Responsible Use

## Purpose

This project is an **educational and research reference implementation** of an
explainable tabular machine-learning pipeline. It demonstrates leakage-safe
preprocessing, model comparison, probability calibration, threshold selection,
evaluation, explainability and serving.

## It is not a medical product

- It is **not** a medical device and has **not** been reviewed or cleared by any
  regulator (FDA, EMA, MHRA, or otherwise).
- Its outputs are **statistical estimates**, not diagnoses. They must never be
  used to diagnose, treat, monitor, or rule out any disease.
- It must not replace, delay, or contradict assessment by a qualified healthcare
  professional.
- It does not provide treatment, medication, or clinical recommendations.

## Data and privacy

- Only public benchmark datasets are used (see `docs/data_dictionary.md`).
- **Never enter real patient information** into the API or dashboard. The
  dashboard examples are synthetic or from public benchmarks.
- The API does not persist request bodies and does not log patient features by
  default. Request identifiers are random and not linked to any record.
- Do not commit datasets that are not licensed for redistribution, secrets, API
  keys, or personally identifiable information.

## Limitations that affect interpretation

- **Representativeness.** The datasets are small, old and demographically narrow:
  the heart data comes from four sites in the 1980s; the diabetes data is a
  symptom questionnaire from a single region; the breast-cancer data reflects one
  clinic's imaging protocol. Benchmark performance is not evidence that the model
  works in another hospital, population, or workflow.
- **Label quality.** Some targets are proxies (e.g. the diabetes label is a
  questionnaire classification, not a laboratory diagnosis).
- **Subgroup performance.** Performance may differ across age, sex and other
  subgroups. This project does **not** claim group fairness; the datasets are too
  small for reliable subgroup estimates, and doing so without sufficient data
  would be misleading. Any subgroup evaluation must be legally and ethically
  appropriate and adequately powered.
- **Explanations.** Feature attributions describe the model, not the biology.
  They are not causal. With correlated features (e.g. the mean/error/worst
  triplets in WDBC) attributions are unstable and can flip between correlated
  inputs.
- **Calibration.** Probabilities are calibrated on small validation splits and
  may not transfer to other populations. Uncalibrated models are labelled as such
  in the API and dashboard.

## Prohibited uses

Do not use this code or its outputs for:

- clinical diagnosis, triage, or treatment decisions;
- patient-facing risk communication without clinician oversight;
- insurance, employment, or other consequential decisions about individuals;
- any use involving real identifiable patient data without appropriate legal
  basis, ethics approval, and safeguards.

## If you build on this

Re-validate on data representative of your target population, quantify subgroup
performance where permissible, monitor for drift, and involve clinicians and
ethicists. Report performance with confidence intervals and calibration, not
accuracy alone.
