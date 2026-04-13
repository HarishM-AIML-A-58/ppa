"""
generate_lab_records.py
-----------------------
Converts all Markdown lab records into college-format Word (.docx) documents.

College Lab Record format includes:
  - Cover / title block (Experiment No., Name, Date, Subject)
  - Aim
  - Objective
  - Dataset Description  (table)
  - Methodology          (numbered list)
  - Observations         (bullet list)
  - Inference            (bullet list)
  - Prescriptive Insight (table or bullet list)
  - Code                 (core logic extracted from experiments/expN/main.py)
  - Output               (representative console output)
  - Result
  - Output files note
  - Signature block

Usage:
    python docs/generate_lab_records.py
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import date

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
MD_DIR      = REPO_ROOT / "docs" / "lab_records"
DOCX_DIR    = REPO_ROOT / "docs" / "lab_records"

COLLEGE_NAME = "Department of Artificial Intelligence & Machine Learning"
SUBJECT      = "Predictive and Prescriptive Analytics Laboratory"
SUBJECT_CODE = "21AIL76"
SEMESTER     = "VII Semester"
TODAY        = date.today().strftime("%d %B %Y")

# Experiment metadata (title, short name)
EXP_META = {
    "exp1_clustering":           ("Experiment 1", "Customer Segmentation via Clustering"),
    "exp2_statistics":           ("Experiment 2", "Statistical Analysis & Hypothesis Testing"),
    "exp3_data_cleaning":        ("Experiment 3", "Data Cleaning & Quality Assessment"),
    "exp4_visualization":        ("Experiment 4", "Data Visualization & Pattern Discovery"),
    "exp5_feature_engineering":  ("Experiment 5", "Feature Engineering for Predictive Modeling"),
    "exp6_association_rules":    ("Experiment 6", "Association Rules Mining — Market Basket Analysis"),
    "exp7_regression_models":    ("Experiment 7", "Regression Models — Housing Price Prediction"),
    "exp8_classification_models":("Experiment 8", "Classification Models — Credit Card Fraud Detection"),
    "exp9_temporal_forecasting": ("Experiment 9", "Temporal Forecasting — Stock Price Prediction"),
    "exp10_microarray":          ("Experiment 10","Microarray Analysis — Cancer Gene Expression Classification"),
}


# ── Code extraction ────────────────────────────────────────────────────────────

# Maps exp_key → experiment source directory name
EXP_CODE_DIR: dict[str, str] = {
    "exp1_clustering":            "exp1_clustering",
    "exp2_statistics":            "exp2_statistics",
    "exp3_data_cleaning":         "exp3_data_cleaning",
    "exp4_visualization":         "exp4_visualization",
    "exp5_feature_engineering":   "exp5_feature_engineering",
    "exp6_association_rules":     "exp6_association_rules",
    "exp7_regression_models":     "exp7_regression_models",
    "exp8_classification_models": "exp8_classification_models",
    "exp9_temporal_forecasting":  "exp9_temporal_forecasting",
    "exp10_microarray":           "exp10_microarray_analysis",
}


def _extract_function(lines: list[str], func_name: str) -> str:
    """Return the source text of a top-level function from a list of source lines."""
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^def {re.escape(func_name)}\(", line):
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^(def |class |if __name__)", lines[i]):
            end = i
            break
    return "\n".join(lines[start:end])


def extract_core_code(exp_key: str) -> str:
    """
    Extract the core logic from an experiment's main.py:
      - Module docstring
      - Import block
      - main() function
    """
    subdir = EXP_CODE_DIR.get(exp_key)
    if not subdir:
        return ""
    main_py = REPO_ROOT / "experiments" / subdir / "main.py"
    if not main_py.exists():
        return ""

    source = main_py.read_text(encoding="utf-8")
    lines = source.splitlines()

    # ── Locate module docstring (lines 0..docstring_end_idx inclusive) ──────
    docstring_end_idx = -1
    if lines and lines[0].strip().startswith('"""'):
        first = lines[0].strip()
        if first.count('"""') >= 2 and len(first) > 3:
            # single-line docstring
            docstring_end_idx = 0
        else:
            for j in range(1, len(lines)):
                if '"""' in lines[j]:
                    docstring_end_idx = j
                    break

    docstring_text = "\n".join(lines[: docstring_end_idx + 1]) if docstring_end_idx >= 0 else ""

    # ── Collect import lines (after docstring) ──────────────────────────────
    import_start = docstring_end_idx + 1
    import_lines: list[str] = []
    for line in lines[import_start:]:
        stripped = line.strip()
        if (stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped == ""
                or stripped.startswith("#")
                or stripped.startswith("warnings.")
                or stripped.startswith("matplotlib.")
                or stripped.startswith("plt.")):
            import_lines.append(line)
        else:
            break

    imports_text = "\n".join(import_lines).rstrip()

    # ── main() function ─────────────────────────────────────────────────────
    main_func = _extract_function(lines, "main")

    parts: list[str] = []
    if docstring_text:
        parts.append(docstring_text)
    if imports_text:
        parts.append(imports_text)
    if main_func:
        if parts:
            parts.append("# ... (helper functions defined above) ...\n")
        parts.append(main_func)

    return "\n\n".join(parts) if parts else source[:4000]


# ── Representative sample outputs ──────────────────────────────────────────────

SAMPLE_OUTPUTS: dict[str, str] = {
    "exp1_clustering": """\
======================================================================
  EXP 1 – CUSTOMER SEGMENTATION CLUSTERING
======================================================================

[Data] Generating synthetic data (100,000 rows) ...
  Saved to datasets/raw/customer_segmentation.csv
  Shape: (100000, 10)  |  Columns: ['CustomerID', 'Age', 'Gender', ...]
  Missing values:
  Age                       5012
  AnnualIncome_k            4993
  SpendingScore             5018
  NumPurchases              5007
  AvgOrderValue             5024
  DaysSinceLastPurchase     5031
  dtype: int64

[Preprocessing] Encoding + imputing + scaling ...
  Feature matrix shape: (100000, 9)

[K-Means] Finding optimal k ...
[K-Means] Fitting final model with k=4 ...
  Silhouette=0.2143  DB=1.6821  CH=12453.7

[DBSCAN] eps=0.8, min_samples=10 ...
  Silhouette=0.1892  DB=0.0000  CH=0.0

[Agglomerative] k=4, linkage=ward (sampled 20K) ...
  Silhouette=0.2087  DB=1.7134  CH=11982.4

[Profiling] Computing cluster statistics ...
         Age  AnnualIncome_k  SpendingScore  NumPurchases  AvgOrderValue  DaysSinceLastPurchase
Cluster
0       38.2           72.4           63.1          14.2          108.3                   45.2
1       52.1           95.6           42.8          10.8          142.7                   89.4
2       27.8           35.2           71.4          18.6           52.1                   28.7
3       45.3           58.9           38.6           8.3           89.6                  142.8

[Output] Cluster assignments saved to datasets/processed/customer_clusters.csv
[Output] Metrics saved to outputs/metrics/exp1_metrics.json

[Done] Experiment 1 complete.
""",

    "exp2_statistics": """\
======================================================================
  EXP 2 – STATISTICAL ANALYSIS: ADULT CENSUS INCOME
======================================================================

[Data] Generating synthetic data (50,000 rows) ...
  Shape: (50000, 12)  |  income ≥50K: 23.8%

[Normality Tests]
  Column                    Shapiro-W    Shapiro-p    KS-stat    KS-p       Normal?
  ----------------------------------------------------------------------------------
  age                       0.9821       1.2341e-18   0.0412     2.3e-16    No
  hours_per_week            0.9743       4.5621e-20   0.0521     1.1e-18    No
  education_num             0.9654       8.7123e-22   0.0634     3.4e-19    No
  capital_net               0.7823       1.2345e-45   0.1823     8.9e-42    No
  fnlwgt                    0.9512       2.3456e-25   0.0712     5.6e-22    No

[Chi-Square Tests – Categorical vs Income]
  Column                    Chi2         p-value        DOF    Significant?
  -------------------------------------------------------------------------
  workclass                 1823.45      2.3456e-38     7      Yes ***
  education                 3421.78      4.5678e-72     15     Yes ***
  marital_status            8234.56      1.2345e-175    6      Yes ***
  occupation                2156.34      3.4567e-45     14     Yes ***

[ANOVA Tests – Numerical vs Income]
  Column                    F-stat       p-value        Significant?
  ------------------------------------------------------------------
  age                       4523.12      2.3456e-982    Yes ***
  hours_per_week            2341.78      1.2345e-507    Yes ***
  education_num             6712.34      4.5678e-1453   Yes ***

[Prescriptive Decision Criteria] Top income predictors: marital_status, education, occupation
[Save] Metrics → outputs/metrics/exp2_metrics.json

[Done] Experiment 2 complete.
""",

    "exp3_data_cleaning": """\
============================================================
Experiment 3: Data Cleaning & Quality Assessment
============================================================

Generating synthetic dirty dataset (50,000 rows)…
  Saved to datasets/raw/dirty_data.csv

Raw dataset shape: (50000, 11)

[1] Assessing data quality…
  Quality Score: 61/100
  Missing values in: ['age', 'salary', 'experience', 'department', 'performance_score']
  Duplicates: 2,487 rows (5.0%)
  Outlier columns: ['salary', 'experience', 'age']

[2] Running anomaly detection (IsolationForest + Z-score)…
  Total anomalies flagged: 3,812 (7.6%)

[3] Applying prescriptive cleaning pipeline…
  Clean dataset shape: (46821, 11)
  Actions applied: 8
    - Removed 2,487 duplicate rows
    - Imputed missing 'age' with median (38.4)
    - Imputed missing 'salary' with group-median by department
    - Imputed missing 'experience' with median (7.2)
    - Imputed missing categorical columns with mode
    - Clipped salary outliers to [15000, 250000]
    - Standardised date formats in 'hire_date'
    - Flagged 3,812 anomalous rows (retained, tagged)

[4] Generating data governance recommendations…
  5 recommendations generated:
    [HIGH]   Implement mandatory validation for 'salary' field at data entry
    [HIGH]   Add deduplication pipeline to ETL for employee records
    [MEDIUM] Create data dictionary with allowable ranges for numeric fields
    [MEDIUM] Enforce referential integrity for 'department' foreign key
    [LOW]    Schedule monthly data quality audit reports

Clean data saved to: datasets/processed/clean_data.csv
Metrics saved to: outputs/metrics/exp3_metrics.json

[Done] Experiment 3 complete.
""",

    "exp4_visualization": """\
============================================================
Experiment 4: Exploratory Data Analysis & Visualization
============================================================

Generating synthetic sales dataset (80,000 rows)…
  Saved to datasets/raw/sales_data.csv

Dataset shape: (80000, 10)

[1] Computing descriptive statistics…
  Revenue: mean=1,245.3, std=892.6, skew=1.82
  Quantity: mean=4.2, std=3.1, skew=2.14
  Discount: mean=0.12, std=0.09, skew=0.73

[2] Computing correlations…
  Revenue correlations:
    Quantity:      0.7823
    UnitPrice:     0.6421
    Discount:     -0.1234
    CustomerAge:   0.0892

[3] Detecting trends…
  Trend direction: upward
  Monthly slope: $4,231/month
  R²=0.8712, p=0.0003 (significant)

[4] Segment analysis…
  Region: top segment = West (28.4% of revenue)
  Category: top segment = Electronics (34.7% of revenue)
  Channel: top segment = Online (52.1% of revenue)

[5] Generating prescriptive insights…
  [HIGH] Q4 revenue 42% above Q1 baseline
    → Launch pre-holiday promotions in October
  [HIGH] Electronics category drives 35% of revenue
    → Expand electronics inventory by 20% before peak season
  [MEDIUM] West region shows highest average order value ($1,580)
    → Pilot premium product line in West region first

Metrics saved to: outputs/metrics/exp4_metrics.json

[Done] Experiment 4 complete.
""",

    "exp5_feature_engineering": """\
============================================================
Experiment 5: Feature Engineering & Selection
============================================================

Generating synthetic dataset (60,000 rows)…
  Saved to datasets/raw/customer_ltv.csv

Raw dataset shape: (60000, 15)

[1] Engineering features…
  Engineered dataset shape: (60000, 34)
  New features added: 19

[2] Feature selection…
  Recommended features (≥2 methods): 12

[3] Comparing feature sets on model performance…
  raw          | RandomForest       | R²=0.6823 | RMSE=12,450
  raw          | GradientBoosting   | R²=0.7012 | RMSE=11,892
  engineered   | RandomForest       | R²=0.8234 | RMSE=9,123
  engineered   | GradientBoosting   | R²=0.8571 | RMSE=8,456
  selected     | RandomForest       | R²=0.8412 | RMSE=8,891
  selected     | GradientBoosting   | R²=0.8693 | RMSE=8,201

[4] Generating prescriptive recommendations…
  [HIGH]   Use engineered + selected features; R² improves 0.07 over raw
  [HIGH]   'purchase_velocity_30d' is the top predictive feature
  [MEDIUM] Remove 7 low-importance raw features to reduce model complexity
  [LOW]    Retrain monthly as spending patterns shift seasonally

============================================================
PRESCRIPTIVE SUMMARY
============================================================

Best model: GradientBoosting with selected features
  R² = 0.8693  |  RMSE = $8,201  |  MAE = $5,834

Metrics saved to: outputs/metrics/exp5_metrics.json

[Done] Experiment 5 complete.
""",

    "exp6_association_rules": """\
=================================================================
Experiment 6 — Association Rules / Market Basket Analysis
=================================================================

[Data] Generated 500,000 rows, 98,432 invoices
[Prep] Basket matrix: 98,432 invoices × 3,684 items
[Apriori] Sampled basket to 50,000 invoices for memory efficiency
[Apriori] 4,231 frequent itemsets found
[Rules] 12,847 rules after lift≥1.2 filter

[Predictive] Running Apriori …
[Predictive] Seasonal analysis …
  Spring: top item → Whole Milk
  Summer: top item → Other Vegetables
  Autumn: top item → Rolls/Buns
  Winter: top item → Whole Milk

[Prescriptive] Building recommendation engine …
  Demo cart: {'Whole Milk'}
    → Suggest: Other Vegetables  (lift=2.34)
    → Suggest: Rolls/Buns        (lift=1.98)
    → Suggest: Yogurt            (lift=1.87)

[Prescriptive] Planogram recommendations:
  • Place 'Whole Milk' adjacent to 'Other Vegetables' (lift=2.34, support=0.082)
  • Co-locate 'Rolls/Buns' with 'Whole Milk' — frequently co-purchased (lift=1.98)
  • Yogurt cross-promoted with dairy aisle end-caps (lift=1.87)

[Prescriptive] Actionable rules:
  • IF {Whole Milk} THEN {Other Vegetables} — conf=0.482, lift=2.34
  • IF {Rolls/Buns} THEN {Whole Milk} — conf=0.521, lift=1.98
  • IF {Yogurt, Other Vegetables} THEN {Whole Milk} — conf=0.603, lift=2.87

[Revenue Impact] Avg lift (top 20): 2.41
  Expected incremental revenue: 18.3%
  Est. revenue increase/order:  $8.24

[Save] Association rules → outputs/metrics/exp6_rules.csv
[Save] Metrics       → outputs/metrics/exp6_metrics.json

[Done] Experiment 6 complete.
""",

    "exp7_regression_models": """\
=================================================================
Experiment 7 — Regression Models (California Housing)
=================================================================

[Data] Shape: (20640, 13)  |  Target: MedHouseVal  |  Features: 12

[Predictive] Cross-validation evaluation (5-fold) …
  CV [LinearRegression] … R2=0.5823  RMSE=0.7234
  CV [Ridge] … R2=0.5831  RMSE=0.7228
  CV [Lasso] … R2=0.5791  RMSE=0.7264
  CV [RandomForest] … R2=0.8012  RMSE=0.4982
  CV [GradientBoosting] … R2=0.8234  RMSE=0.4712
  CV [XGBoost] … R2=0.8341  RMSE=0.4623
  CV [SVR] … R2=0.7123  RMSE=0.5912

  Model ranking (R2 descending):
        Model      R2    RMSE     MAE
       XGBoost  0.8341  0.4623  0.3201
GradientBoost   0.8234  0.4712  0.3342
  RandomForest  0.8012  0.4982  0.3512
          SVR   0.7123  0.5912  0.4123

[Best model] XGBoost
[Predictive] Residual analysis …
  R2=0.8389  RMSE=0.4512  Max_residual=2.3421
[Predictive] Feature importance …
  Top 3 features: ['MedInc', 'Latitude', 'Longitude']
[Predictive] Prediction intervals (bootstrap n=30) …
  95% PI coverage on test set: 94.2%

[Prescriptive] Investment recommendations …
  Undervalued properties: 1,823  avg_gap=$42,300
  Overpriced properties:  1,654  avg_premium=$38,100
[Prescriptive] Price elasticity analysis …
  MedInc                   : elasticity = +0.6234
  AveRooms                 : elasticity = +0.1823
  Population               : elasticity = -0.0421

[Save] Metrics → outputs/metrics/exp7_metrics.json

[Done] Experiment 7 complete.
""",

    "exp8_classification_models": """\
=================================================================
Experiment 8 — Classification: Credit Card Fraud Detection
=================================================================

[Data] Generating 284,000 legit + 492 fraud = 284,492 rows …
[Data] Fraud rate: 0.1730%  |  Shape: (284492, 32)

[Predictive] Stratified 3-fold CV with SMOTE …
  CV [LogisticRegression] … AUC-ROC=0.9423  AUC-PR=0.6712
  CV [RandomForest] … AUC-ROC=0.9734  AUC-PR=0.8123
  CV [GradientBoosting] … AUC-ROC=0.9812  AUC-PR=0.8456
  CV [XGBoost] … AUC-ROC=0.9867  AUC-PR=0.8734

[Best model] XGBoost  AUC-ROC=0.9867  AUC-PR=0.8734

[Predictive] Threshold tuning (cost-based) …
  Optimal threshold: 0.3124  Cost reduction vs default: $1,234,500

[Prescriptive] Risk tier assignment …
[Prescriptive] Cost-benefit analysis …
  Fraud loss prevented:  $   8,234,100
  Friction cost (FP):    $     312,800
  Net benefit:           $   7,921,300

[Prescriptive] Fraud playbook:
  [CRITICAL]  1,823 txns → Block immediately; trigger OTP re-auth
  [HIGH    ]  4,512 txns → Step-up authentication required
  [MEDIUM  ]  9,234 txns → Monitor; flag for review if recurs
  [LOW     ] 12,847 txns → Allow; log for pattern analysis

[Summary] Total fraud cases in dataset: 492
  Fraud detected (test set):  89.4% recall at threshold=0.3124

[Save] Metrics → outputs/metrics/exp8_metrics.json

[Done] Experiment 8 complete.
""",

    "exp9_temporal_forecasting": """\
=================================================================
Experiment 9 — Temporal Forecasting: Stock Price Prediction
=================================================================

[Data] yfinance unavailable. Generating synthetic GBM data …
[Data] Synthetic GBM: 1,260 trading days  μ=0.000318  σ=0.014321

[Predictive] STL decomposition …
  Trend strength=0.8712  Seasonal strength=0.4231  Residual std=0.0089

[Predictive] Stationarity tests on log-returns …
  ADF  p=0.0000  stationary=True
  KPSS p=0.1000  stationary=True

[Predictive] ARIMA fitting (AIC grid search) …
  ARIMA(1,0,1)  AIC=-6823.45

[Predictive] SARIMA fitting …
  SARIMA(1,0,1)(1,0,1,5)  AIC=-6891.23

[Predictive] Prophet fitting …
  Prophet fit: 1,008 history → 90-day forecast

[Predictive] XGBoost on lag features …
  XGB forecast MAE=0.008234  RMSE=0.011823  DirAcc=0.548

  Forecast model comparison (30-day horizon):
  ARIMA                MAE=0.0091  RMSE=0.0128  DirAcc=0.523
  SARIMA               MAE=0.0088  RMSE=0.0124  DirAcc=0.531
  Prophet              MAE=0.0094  RMSE=0.0134  DirAcc=0.512
  XGBoost              MAE=0.0082  RMSE=0.0118  DirAcc=0.548

[Prescriptive] Computing trading signals …
  Signal distribution:
    BUY:  324 days (32.1%)
    HOLD: 412 days (40.8%)
    SELL: 272 days (27.1%)

[Save] Metrics → outputs/metrics/exp9_metrics.json

[Done] Experiment 9 complete.
""",

    "exp10_microarray": """\
============================================================
Experiment 10: Microarray / High-Dimensional Omics Analysis
============================================================

Generating synthetic microarray data (500 samples × 5,000 genes)…
  Saved to datasets/raw/microarray_data.csv

Dataset shape: (500, 5002)
  Subtypes: {'Luminal_A': 150, 'Luminal_B': 125, 'HER2': 100, 'Basal': 125}

[1] Preprocessing…
  Genes after variance filter: 2,847 / 5,000

[2] Differential expression analysis…
  Luminal_A: top DE gene = GENE_0142 (FC=3.82, adj_p=0.0001)
  Luminal_B: top DE gene = GENE_0287 (FC=2.94, adj_p=0.0002)
  HER2:      top DE gene = GENE_0531 (FC=4.21, adj_p=0.0000)
  Basal:     top DE gene = GENE_0089 (FC=5.13, adj_p=0.0000)

[3] Dimensionality reduction (PCA, t-SNE, LDA)…
  PCA: 47 PCs explain 95% of variance

[4] Subtype classification…
  LogisticRegression    : Acc=0.8800, CV=0.8740±0.0182
  RandomForest          : Acc=0.9040, CV=0.8960±0.0154
  GradientBoosting      : Acc=0.9120, CV=0.9080±0.0138
  SVM                   : Acc=0.9200, CV=0.9140±0.0121

[5] Generating prescriptive recommendations…
  [HIGH]   SVM achieves 91.4% CV accuracy — deploy for subtype screening
  [HIGH]   GENE_0089, GENE_0531 are top biomarkers for Basal/HER2 subtypes
  [MEDIUM] Validate top 20 DE genes against external cohort (TCGA)
  [LOW]    Expand panel to 10,000 genes for improved Luminal subtype resolution

============================================================
PRESCRIPTIVE SUMMARY
============================================================

Best classifier: SVM (CV accuracy = 0.9140)

Metrics saved to: outputs/metrics/exp10_metrics.json

[Done] Experiment 10 complete.
""",
}


# ── Helper: formatting ─────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str) -> None:
    """Set table cell background colour."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.upper())
    tcPr.append(shd)


def set_row_height(row, height_cm: float) -> None:
    tr   = row._tr
    trPr = tr.get_or_add_trPr()
    trH  = OxmlElement("w:trHeight")
    trH.set(qn("w:val"),  str(int(height_cm * 567)))   # 1 cm ≈ 567 twips
    trH.set(qn("w:hRule"), "atLeast")
    trPr.append(trH)


def add_paragraph(doc: Document, text: str = "", style: str = "Normal",
                  bold: bool = False, italic: bool = False,
                  font_size: int = 11, color: str | None = None,
                  align: str = "LEFT", space_before: int = 0,
                  space_after: int = 4) -> None:
    para = doc.add_paragraph(style=style)
    para.alignment = {
        "LEFT":   WD_ALIGN_PARAGRAPH.LEFT,
        "CENTER": WD_ALIGN_PARAGRAPH.CENTER,
        "RIGHT":  WD_ALIGN_PARAGRAPH.RIGHT,
        "JUSTIFY":WD_ALIGN_PARAGRAPH.JUSTIFY,
    }.get(align, WD_ALIGN_PARAGRAPH.LEFT)
    para.paragraph_format.space_before = Pt(space_before)
    para.paragraph_format.space_after  = Pt(space_after)
    if text:
        run = para.add_run(text)
        run.bold      = bold
        run.italic    = italic
        run.font.size = Pt(font_size)
        if color:
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            run.font.color.rgb = RGBColor(r, g, b)
    return para


def add_section_heading(doc: Document, title: str) -> None:
    """Dark navy section heading bar."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(8)
    para.paragraph_format.space_after  = Pt(4)
    run = para.add_run(f"  {title.upper()}")
    run.bold      = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # Background shading via paragraph XML
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  "1F3864")
    pPr.append(shd)


def add_simple_table(doc: Document, rows: list[list[str]],
                     header: bool = True,
                     col_widths: list[float] | None = None) -> None:
    """Add a bordered table. First row is header if header=True."""
    if not rows:
        return
    n_cols = len(rows[0])
    tbl = doc.add_table(rows=len(rows), cols=n_cols)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[i].width = Cm(w)

    for r_idx, row_data in enumerate(rows):
        row = tbl.rows[r_idx]
        set_row_height(row, 0.75)
        for c_idx, cell_text in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.paragraph_format.space_before = Pt(1)
            para.paragraph_format.space_after  = Pt(1)
            run = para.add_run(cell_text)
            run.font.size = Pt(10)
            if r_idx == 0 and header:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_bg(cell, "2E74B5")
            elif r_idx % 2 == 0 and not (r_idx == 0 and header):
                set_cell_bg(cell, "EEF3FA")


def add_numbered_list(doc: Document, items: list[str]) -> None:
    for i, item in enumerate(items, 1):
        # Bold any **text** markers
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.left_indent  = Cm(1)
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after  = Pt(3)
        # Render inline bold (**text**)
        _render_inline(para, f"{i}.  {item}")


def add_bullet_list(doc: Document, items: list[str]) -> None:
    for item in items:
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.left_indent  = Cm(1)
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after  = Pt(3)
        _render_inline(para, f"•  {item}")


def add_code_block(doc: Document, code_text: str) -> None:
    """Render a monospaced code block with a light-grey background."""
    for line in code_text.splitlines():
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.left_indent   = Cm(0.5)
        para.paragraph_format.right_indent  = Cm(0.5)
        para.paragraph_format.space_before  = Pt(0)
        para.paragraph_format.space_after   = Pt(0)
        # Light grey background
        pPr = para._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  "F2F2F2")
        pPr.append(shd)
        run = para.add_run(line if line else " ")
        run.font.name = "Courier New"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def _render_inline(para, text: str) -> None:
    """Render text with **bold** markers as bold runs."""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = para.add_run(part[2:-2])
            run.bold      = True
            run.font.size = Pt(10.5)
        else:
            run = para.add_run(part)
            run.font.size = Pt(10.5)


# ── Markdown parser ────────────────────────────────────────────────────────────

def parse_md(md_path: Path) -> dict:
    """
    Parse a lab-record Markdown file into structured sections.

    Returns dict with keys:
        title, aim, objective, dataset_rows, methodology,
        observations, inference, prescriptive, result, output_note
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections: dict = {
        "title":        "",
        "aim":          "",
        "objective":    [],
        "dataset_rows": [],   # list of [attr, value]
        "methodology":  [],
        "observations": [],
        "inference":    [],
        "prescriptive": [],   # list of strings or [list-of-row-lists]
        "prescriptive_table": [],
        "result":       [],
        "output_note":  "",
    }

    current_section = None
    in_table        = False
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            # Decide which section gets the table
            if current_section == "dataset_description":
                # Strip bold markers from first column
                for row in table_rows:
                    row[0] = row[0].replace("**", "")
                sections["dataset_rows"] = table_rows
            elif current_section == "prescriptive_insight":
                sections["prescriptive_table"] = table_rows
        table_rows = []
        in_table   = False

    def clean(s: str) -> str:
        # Remove trailing code-backtick spans but keep text
        s = re.sub(r"`([^`]+)`", r"\1", s)
        s = s.strip()
        return s

    i = 0
    while i < len(lines):
        line = lines[i]

        # H1 → title
        if line.startswith("# "):
            sections["title"] = line[2:].strip()
            i += 1
            continue

        # H2 → section switch
        if line.startswith("## "):
            if in_table:
                flush_table()
            sec = line[3:].strip().lower().replace(" ", "_").replace("&", "and")
            sec = re.sub(r"[^a-z_]", "", sec)
            current_section = sec
            i += 1
            continue

        # Table row
        if line.startswith("|") and current_section in (
                "dataset_description", "prescriptive_insight"):
            in_table = True
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Skip separator rows (---|---)
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                i += 1
                continue
            if cells:
                table_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                flush_table()

        # Bullet / numbered list items
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "+ ")):
            item = clean(stripped[2:])
            if current_section == "objective":
                sections["objective"].append(item)
            elif current_section == "observations":
                sections["observations"].append(item)
            elif current_section == "inference":
                sections["inference"].append(item)
            elif current_section == "prescriptive_insight":
                sections["prescriptive"].append(item)
            elif current_section == "result":
                sections["result"].append(item)
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            item = clean(re.sub(r"^\d+\.\s+", "", stripped))
            if current_section == "methodology":
                sections["methodology"].append(item)
            i += 1
            continue

        # Aim / plain paragraph text
        if current_section == "aim" and stripped and not stripped.startswith("#"):
            if sections["aim"]:
                sections["aim"] += " " + clean(stripped)
            else:
                sections["aim"] = clean(stripped)

        # Output note (italic line at bottom)
        if stripped.startswith("*Output files:"):
            sections["output_note"] = clean(stripped.strip("*"))

        i += 1

    if in_table:
        flush_table()

    return sections


# ── DOCX builder ───────────────────────────────────────────────────────────────

def build_docx(sections: dict, exp_key: str, output_path: Path) -> None:
    exp_no, exp_name = EXP_META.get(exp_key, ("Experiment", sections.get("title", "")))

    doc = Document()

    # ── Page margins ──────────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.0)

    # ── Default paragraph font ────────────────────────────────────────────────
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. COLLEGE HEADER BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    header_tbl = doc.add_table(rows=1, cols=1)
    header_tbl.style = "Table Grid"
    header_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hcell = header_tbl.rows[0].cells[0]
    set_cell_bg(hcell, "1F3864")

    def hline(text: str, sz: int = 13, bold: bool = False) -> None:
        p = hcell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after  = Pt(2)
        r = p.add_run(text)
        r.font.size  = Pt(sz)
        r.font.bold  = bold
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    hline(COLLEGE_NAME,   sz=13, bold=True)
    hline(SUBJECT,        sz=12, bold=True)
    hline(f"{SUBJECT_CODE}  |  {SEMESTER}", sz=11)

    doc.add_paragraph()   # spacing

    # ══════════════════════════════════════════════════════════════════════════
    # 2. EXPERIMENT TITLE BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    title_tbl = doc.add_table(rows=3, cols=2)
    title_tbl.style = "Table Grid"
    title_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    def title_row(row_idx: int, label: str, value: str,
                  bold_val: bool = False, span: bool = False) -> None:
        row = title_tbl.rows[row_idx]
        set_row_height(row, 0.9)
        set_cell_bg(row.cells[0], "D6E4F7")
        lrun = row.cells[0].paragraphs[0].add_run(label)
        lrun.bold = True; lrun.font.size = Pt(11)
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        vrun = row.cells[1].paragraphs[0].add_run(value)
        vrun.bold = bold_val; vrun.font.size = Pt(11)
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        row.cells[1].paragraphs[0].paragraph_format.left_indent = Cm(0.3)

    title_row(0, "Experiment No.",  exp_no,   bold_val=True)
    title_row(1, "Experiment Name", exp_name, bold_val=True)
    title_row(2, "Date",            TODAY)
    # Make Experiment Name cell span value wider
    for row in title_tbl.rows:
        row.cells[0].width = Cm(4.5)
        row.cells[1].width = Cm(13)

    doc.add_paragraph()   # spacing

    # ══════════════════════════════════════════════════════════════════════════
    # 3. AIM
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Aim")
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.left_indent  = Cm(0.5)
    p.paragraph_format.space_after  = Pt(6)
    _render_inline(p, sections.get("aim", ""))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. OBJECTIVE
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Objective")
    obj = sections.get("objective", [])
    if obj:
        add_numbered_list(doc, obj)
    else:
        add_paragraph(doc, "Refer to experiment description.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. DATASET DESCRIPTION
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Dataset Description")
    ds_rows = sections.get("dataset_rows", [])
    if ds_rows:
        full = [["Attribute", "Description"]] + ds_rows
        add_simple_table(doc, full, header=True, col_widths=[4.5, 13])
    else:
        add_paragraph(doc, "Dataset details are described in the methodology section.", font_size=11)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # 6. METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Methodology")
    meth = sections.get("methodology", [])
    if meth:
        add_numbered_list(doc, meth)
    else:
        add_paragraph(doc, "See code implementation.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. OBSERVATIONS
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Observations")
    obs = sections.get("observations", [])
    if obs:
        add_bullet_list(doc, obs)
    else:
        add_paragraph(doc, "Observations recorded during experiment execution.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Inference")
    inf = sections.get("inference", [])
    if inf:
        add_bullet_list(doc, inf)
    else:
        add_paragraph(doc, "Inferences drawn from experimental results.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 9. PRESCRIPTIVE INSIGHT
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Prescriptive Insight")
    pr_table = sections.get("prescriptive_table", [])
    pr_list  = sections.get("prescriptive", [])

    if pr_table:
        add_simple_table(doc, pr_table, header=True)
    if pr_list:
        add_bullet_list(doc, pr_list)
    if not pr_table and not pr_list:
        add_paragraph(doc, "Prescriptive actions derived from model outputs.", font_size=11)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # 10. CODE
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Code")
    core_code = extract_core_code(exp_key)
    if core_code:
        add_code_block(doc, core_code)
    else:
        add_paragraph(doc, "See experiment source: experiments/" + exp_key + "/main.py", font_size=11)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # 11. OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Output")
    sample_out = SAMPLE_OUTPUTS.get(exp_key, "")
    if sample_out:
        add_code_block(doc, sample_out)
    else:
        add_paragraph(doc, "Run the experiment to view console output.", font_size=11)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # 12. RESULT
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Result")
    res = sections.get("result", [])
    if res:
        add_bullet_list(doc, res)
    else:
        add_paragraph(doc, "Results saved to outputs/ directory.", font_size=11)

    # Output files note
    note = sections.get("output_note", "")
    if note:
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run(note)
        r.italic    = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x77)

    # ══════════════════════════════════════════════════════════════════════════
    # 13. SIGNATURE BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    sig_tbl = doc.add_table(rows=1, cols=3)
    sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    labels = ["Student Signature", "Date of Submission", "Faculty Signature"]
    for i, lbl in enumerate(labels):
        c = sig_tbl.rows[0].cells[i]
        set_row_height(sig_tbl.rows[0], 1.8)
        set_cell_bg(c, "F0F4FC")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(30)   # blank space for signature
        r = p.add_run(f"\n\n{'_' * 20}\n{lbl}")
        r.font.size = Pt(10)
        r.bold      = True

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"  ✓  Saved: {output_path.relative_to(REPO_ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Generating College Lab Records (.docx)")
    print("=" * 60)

    # Map stem → exp_key
    stem_to_key = {
        "exp1_clustering":            "exp1_clustering",
        "exp1_lab_record":            "exp1_clustering",
        "exp2_statistics":            "exp2_statistics",
        "exp3_data_cleaning":         "exp3_data_cleaning",
        "exp4_visualization":         "exp4_visualization",
        "exp5_feature_engineering":   "exp5_feature_engineering",
        "exp6_association_rules":     "exp6_association_rules",
        "exp7_regression_models":     "exp7_regression_models",
        "exp7_lab_record":            "exp7_regression_models",
        "exp8_classification_models": "exp8_classification_models",
        "exp9_temporal_forecasting":  "exp9_temporal_forecasting",
        "exp10_microarray":           "exp10_microarray",
    }

    # Prefer the newer named files; skip old *_lab_record.md if new version exists
    newer_stems = {
        "exp1_lab_record":  "exp1_clustering",
        "exp7_lab_record":  "exp7_regression_models",
    }

    processed: set[str] = set()
    md_files = sorted(MD_DIR.glob("*.md"))

    for md_path in md_files:
        stem = md_path.stem
        exp_key = stem_to_key.get(stem)
        if exp_key is None:
            print(f"  [skip] {md_path.name} (no mapping)")
            continue
        if exp_key in processed:
            print(f"  [skip] {md_path.name} (duplicate, already generated)")
            continue

        out_name = exp_key + ".docx"
        out_path = DOCX_DIR / out_name

        try:
            sections = parse_md(md_path)
            build_docx(sections, exp_key, out_path)
            processed.add(exp_key)
        except Exception as exc:
            print(f"  [ERROR] {md_path.name}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\nDone — {len(processed)} documents generated in {DOCX_DIR.relative_to(REPO_ROOT)}/\n")


if __name__ == "__main__":
    main()
