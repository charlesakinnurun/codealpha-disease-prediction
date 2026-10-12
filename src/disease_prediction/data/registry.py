"""Dataset registry: provenance, licence, schema and target definition.

Every dataset is documented here with its official source, licence, citation and
the target convention actually used by this project. Nothing is downloaded or
substituted implicitly — :func:`get_schema` and the CLI tools are the single
source of truth.
"""

from __future__ import annotations

from disease_prediction.config import TASKS
from disease_prediction.data.schemas import DatasetSchema
from disease_prediction.data.schemas import FeatureKind as K
from disease_prediction.data.schemas import make_feature as _f

# ---------------------------------------------------------------------------
# Heart Disease — UCI Cleveland subset (14 attributes, 303 instances)
# ---------------------------------------------------------------------------
_HEART = DatasetSchema(
    task="heart_disease",
    display_name="Heart Disease (UCI Cleveland)",
    source_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
    license="CC BY 4.0 (UCI Machine Learning Repository)",
    citation=(
        "Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. (1988). "
        "Heart Disease. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C52P4X"
    ),
    target_column="num",
    positive_class=1,
    negative_class=0,
    target_meaning="presence of angiographic coronary artery disease",
    target_description=(
        "Original 'num' is integer 0-4 (0 = no disease, 1-4 = increasing "
        "severity). We binarise to presence (num > 0)."
    ),
    features=(
        _f("age", K.NUMERIC, "Age in years", unit="years", minimum=18, maximum=120),
        _f("sex", K.BINARY, "Biological sex as recorded", allowed_values=(0, 1)),
        _f(
            "cp",
            K.CATEGORICAL,
            "Chest pain type",
            allowed_values=(1, 2, 3, 4),
        ),
        _f(
            "trestbps",
            K.NUMERIC,
            "Resting blood pressure on admission",
            unit="mm Hg",
            minimum=80,
            maximum=220,
        ),
        _f(
            "chol",
            K.NUMERIC,
            "Serum cholesterol",
            unit="mg/dl",
            minimum=100,
            maximum=650,
        ),
        _f("fbs", K.BINARY, "Fasting blood sugar > 120 mg/dl", allowed_values=(0, 1)),
        _f(
            "restecg",
            K.CATEGORICAL,
            "Resting electrocardiographic result",
            allowed_values=(0, 1, 2),
        ),
        _f(
            "thalach",
            K.NUMERIC,
            "Maximum heart rate achieved during exercise test",
            unit="bpm",
            minimum=60,
            maximum=220,
        ),
        _f("exang", K.BINARY, "Exercise-induced angina", allowed_values=(0, 1)),
        _f(
            "oldpeak",
            K.NUMERIC,
            "ST depression induced by exercise relative to rest",
            unit="mm",
            minimum=-10.0,
            maximum=10.0,
        ),
        _f(
            "slope",
            K.CATEGORICAL,
            "Slope of the peak exercise ST segment",
            allowed_values=(1, 2, 3),
        ),
        _f(
            "ca",
            K.CATEGORICAL,
            "Number of major vessels coloured by fluoroscopy",
            allowed_values=(0, 1, 2, 3),
            optional=True,
        ),
        _f(
            "thal",
            K.CATEGORICAL,
            "Thalassemia / thallium stress result code",
            allowed_values=(3, 6, 7),
            optional=True,
        ),
    ),
    notes=(
        "Raw file processed.cleveland.data has no header; missing values are "
        "encoded as '?'. The publication request in heart-disease.names asks "
        "that the principal investigators be named in resulting publications."
    ),
)

# ---------------------------------------------------------------------------
# Diabetes — UCI Early Stage Diabetes Risk Prediction (520 instances)
# ---------------------------------------------------------------------------
_DIABETES_YES_NO = (0, 1)
_DIABETES_SYMPTOMS = (
    ("Polyuria", "Excessive urination"),
    ("Polydipsia", "Excessive thirst"),
    ("sudden_weight_loss", "Sudden unexplained weight loss"),
    ("weakness", "Generalised weakness"),
    ("Polyphagia", "Excessive hunger"),
    ("Genital_thrush", "Genital thrush infection"),
    ("visual_blurring", "Visual blurring"),
    ("Itching", "Itching"),
    ("Irritability", "Irritability"),
    ("delayed_healing", "Delayed wound healing"),
    ("partial_paresis", "Partial paresis (muscle weakness)"),
    ("muscle_stiffness", "Muscle stiffness"),
    ("Alopecia", "Alopecia (hair loss)"),
    ("Obesity", "Obesity"),
)

_DIABETES = DatasetSchema(
    task="diabetes",
    display_name="Early Stage Diabetes Risk Prediction (UCI)",
    source_url="https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset",
    license="CC BY 4.0 (UCI Machine Learning Repository)",
    citation=(
        "Islam, M. M. F., Ferdousi, R., Rahman, S., Bushra, H. Y. (2020). "
        "Likelihood Prediction of Diabetes at Early Stage Using Data Mining "
        "Techniques. In Computer Vision and Machine Intelligence in Medical "
        "Image Analysis. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5VG8H"
    ),
    target_column="class",
    positive_class="Positive",
    negative_class="Negative",
    target_meaning="screening-classified diabetes risk (Positive/Negative)",
    target_description=(
        "Symptom-based questionnaire label. This is NOT a laboratory "
        "diagnosis of diabetes; it is an early-stage risk screening label."
    ),
    features=(
        _f("Age", K.NUMERIC, "Age in years", unit="years", minimum=16, maximum=100),
        _f("Gender", K.CATEGORICAL, "Recorded gender", allowed_values=("Male", "Female")),
        *[
            _f(name, K.BINARY, desc, allowed_values=_DIABETES_YES_NO)
            for name, desc in _DIABETES_SYMPTOMS
        ],
    ),
    notes=(
        "Symptom questionnaire from Sylhet, Bangladesh; the sample is small "
        "(520 rows) and demographically narrow. Labels derive from a "
        "questionnaire rather than clinical testing, so this is a risk-signal "
        "model only."
    ),
)

# ---------------------------------------------------------------------------
# Breast Cancer — UCI Breast Cancer Wisconsin (Diagnostic), via scikit-learn
# ---------------------------------------------------------------------------
_BC_MEAN = (
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
)
_BC_ERROR = (
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
)
_BC_WORST = (
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
)

_BC_FEATURES = tuple(
    _f(name, K.NUMERIC, f"Cell-nuclei measurement: {name}", minimum=0.0)
    for name in (*_BC_MEAN, *_BC_ERROR, *_BC_WORST)
)

_BREAST_CANCER = DatasetSchema(
    task="breast_cancer",
    display_name="Breast Cancer Wisconsin (Diagnostic) (UCI)",
    source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    license="CC BY 4.0 (UCI Machine Learning Repository) / bundled with scikit-learn (BSD-3-Clause)",
    citation=(
        "Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). Nuclear "
        "feature extraction for breast tumor diagnosis. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5DW2B"
    ),
    target_column="target",
    positive_class=1,
    negative_class=0,
    target_meaning="malignant breast mass",
    target_description=(
        "scikit-learn exposes the target as 0 = malignant, 1 = benign. We "
        "document the positive class as *malignant* (target == 0) and store a "
        "'malignant' indicator whether the model was trained as such."
    ),
    features=_BC_FEATURES,
    notes=(
        "Loaded from scikit-learn's bundled copy of the UCI WDBC dataset, so "
        "no network access is required. Features are computed from digitised "
        "fine-needle aspirate images and are highly correlated within the "
        "mean/error/worst triplets, which makes single-feature attribution "
        "unstable."
    ),
)

_REGISTRY: dict[str, DatasetSchema] = {
    "heart_disease": _HEART,
    "diabetes": _DIABETES,
    "breast_cancer": _BREAST_CANCER,
}


def get_schema(task: str) -> DatasetSchema:
    """Return the schema for a task, raising a clear error for unknown tasks."""
    if task not in _REGISTRY:
        raise KeyError(f"Unknown task {task!r}. Available tasks: {', '.join(TASKS)}")
    return _REGISTRY[task]


def all_schemas() -> dict[str, DatasetSchema]:
    """Return a copy of the full registry."""
    return dict(_REGISTRY)
