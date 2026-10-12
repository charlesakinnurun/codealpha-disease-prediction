# Data Dictionary

> Generated from `disease_prediction.data.registry` by `scripts/generate_docs.py`. Do not edit by hand.

Feature bounds are inclusive plausibility limits used for validation; they are wider than the observed dataset range on purpose. `optional` features may contain missing values, which the training pipeline imputes.

## Heart Disease (UCI Cleveland)

- **Task id:** `heart_disease`
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Licence:** CC BY 4.0 (UCI Machine Learning Repository)
- **Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- **Target column (raw):** `num`
- **Target meaning:** presence of angiographic coronary artery disease
- **Target definition:** Original 'num' is integer 0-4 (0 = no disease, 1-4 = increasing severity). We binarise to presence (num > 0).

| Feature | Kind | Unit | Allowed / Range | Optional | Description |
| --- | --- | --- | --- | :---: | --- |
| `age` | numeric | years | [18, 120] | no | Age in years |
| `sex` | binary | — | 0, 1 | no | Biological sex as recorded |
| `cp` | categorical | — | 1, 2, 3, 4 | no | Chest pain type |
| `trestbps` | numeric | mm Hg | [80, 220] | no | Resting blood pressure on admission |
| `chol` | numeric | mg/dl | [100, 650] | no | Serum cholesterol |
| `fbs` | binary | — | 0, 1 | no | Fasting blood sugar > 120 mg/dl |
| `restecg` | categorical | — | 0, 1, 2 | no | Resting electrocardiographic result |
| `thalach` | numeric | bpm | [60, 220] | no | Maximum heart rate achieved during exercise test |
| `exang` | binary | — | 0, 1 | no | Exercise-induced angina |
| `oldpeak` | numeric | mm | [-10.0, 10.0] | no | ST depression induced by exercise relative to rest |
| `slope` | categorical | — | 1, 2, 3 | no | Slope of the peak exercise ST segment |
| `ca` | categorical | — | 0, 1, 2, 3 | yes | Number of major vessels coloured by fluoroscopy |
| `thal` | categorical | — | 3, 6, 7 | yes | Thalassemia / thallium stress result code |

**Notes:** Raw file processed.cleveland.data has no header; missing values are encoded as '?'. The publication request in heart-disease.names asks that the principal investigators be named in resulting publications.

## Early Stage Diabetes Risk Prediction (UCI)

- **Task id:** `diabetes`
- **Source:** https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
- **Licence:** CC BY 4.0 (UCI Machine Learning Repository)
- **Citation:** Islam, M. M. F., Ferdousi, R., Rahman, S., Bushra, H. Y. (2020). Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques. In Computer Vision and Machine Intelligence in Medical Image Analysis. UCI Machine Learning Repository. https://doi.org/10.24432/C5VG8H
- **Target column (raw):** `class`
- **Target meaning:** screening-classified diabetes risk (Positive/Negative)
- **Target definition:** Symptom-based questionnaire label. This is NOT a laboratory diagnosis of diabetes; it is an early-stage risk screening label.

| Feature | Kind | Unit | Allowed / Range | Optional | Description |
| --- | --- | --- | --- | :---: | --- |
| `Age` | numeric | years | [16, 100] | no | Age in years |
| `Gender` | categorical | — | Male, Female | no | Recorded gender |
| `Polyuria` | binary | — | 0, 1 | no | Excessive urination |
| `Polydipsia` | binary | — | 0, 1 | no | Excessive thirst |
| `sudden_weight_loss` | binary | — | 0, 1 | no | Sudden unexplained weight loss |
| `weakness` | binary | — | 0, 1 | no | Generalised weakness |
| `Polyphagia` | binary | — | 0, 1 | no | Excessive hunger |
| `Genital_thrush` | binary | — | 0, 1 | no | Genital thrush infection |
| `visual_blurring` | binary | — | 0, 1 | no | Visual blurring |
| `Itching` | binary | — | 0, 1 | no | Itching |
| `Irritability` | binary | — | 0, 1 | no | Irritability |
| `delayed_healing` | binary | — | 0, 1 | no | Delayed wound healing |
| `partial_paresis` | binary | — | 0, 1 | no | Partial paresis (muscle weakness) |
| `muscle_stiffness` | binary | — | 0, 1 | no | Muscle stiffness |
| `Alopecia` | binary | — | 0, 1 | no | Alopecia (hair loss) |
| `Obesity` | binary | — | 0, 1 | no | Obesity |

**Notes:** Symptom questionnaire from Sylhet, Bangladesh; the sample is small (520 rows) and demographically narrow. Labels derive from a questionnaire rather than clinical testing, so this is a risk-signal model only.

## Breast Cancer Wisconsin (Diagnostic) (UCI)

- **Task id:** `breast_cancer`
- **Source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Licence:** CC BY 4.0 (UCI Machine Learning Repository) / bundled with scikit-learn (BSD-3-Clause)
- **Citation:** Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- **Target column (raw):** `target`
- **Target meaning:** malignant breast mass
- **Target definition:** scikit-learn exposes the target as 0 = malignant, 1 = benign. We document the positive class as *malignant* (target == 0) and store a 'malignant' indicator whether the model was trained as such.

| Feature | Kind | Unit | Allowed / Range | Optional | Description |
| --- | --- | --- | --- | :---: | --- |
| `mean radius` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean radius |
| `mean texture` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean texture |
| `mean perimeter` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean perimeter |
| `mean area` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean area |
| `mean smoothness` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean smoothness |
| `mean compactness` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean compactness |
| `mean concavity` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean concavity |
| `mean concave points` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean concave points |
| `mean symmetry` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean symmetry |
| `mean fractal dimension` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: mean fractal dimension |
| `radius error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: radius error |
| `texture error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: texture error |
| `perimeter error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: perimeter error |
| `area error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: area error |
| `smoothness error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: smoothness error |
| `compactness error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: compactness error |
| `concavity error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: concavity error |
| `concave points error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: concave points error |
| `symmetry error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: symmetry error |
| `fractal dimension error` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: fractal dimension error |
| `worst radius` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst radius |
| `worst texture` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst texture |
| `worst perimeter` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst perimeter |
| `worst area` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst area |
| `worst smoothness` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst smoothness |
| `worst compactness` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst compactness |
| `worst concavity` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst concavity |
| `worst concave points` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst concave points |
| `worst symmetry` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst symmetry |
| `worst fractal dimension` | numeric | — | [0.0, None] | no | Cell-nuclei measurement: worst fractal dimension |

**Notes:** Loaded from scikit-learn's bundled copy of the UCI WDBC dataset, so no network access is required. Features are computed from digitised fine-needle aspirate images and are highly correlated within the mean/error/worst triplets, which makes single-feature attribution unstable.
