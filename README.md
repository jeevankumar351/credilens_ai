# 🏦 CrediLens AI — Intelligent Loan Approval System

A bank-grade, end-to-end ML system that predicts loan approval, generates a
risk probability score, explains every decision using SHAP, and presents it
all in a professional fintech-style Streamlit dashboard.

---

## 📁 Project Structure

```
credilens_ai/
├── generate_dataset.py      # Creates the structured loan dataset
├── preprocessing.py         # Cleaning, encoding, scaling, feature engineering
├── model_training.py        # Trains & compares 4 ML models, runs SHAP, saves artifacts
├── app.py                   # Streamlit banking dashboard (main web app)
├── pdf_report.py            # Generates downloadable PDF decision reports
├── colab_launcher.py        # One-command launcher for Google Colab
├── requirements.txt         # All Python dependencies
├── data/
│   └── loan_data.csv        # Generated dataset
├── models/
│   ├── trained_model.pkl    # Best model (selected by ROC-AUC)
│   ├── preprocessor.pkl     # Fitted LoanPreprocessor (encoders + scaler)
│   ├── shap_explainer.pkl   # SHAP explainer for live, per-prediction reasoning
│   └── metadata.json        # Model metrics, comparison table, feature importance
└── assets/
    ├── approval_distribution.png
    ├── income_vs_approval.png
    ├── credit_history_vs_approval.png
    ├── property_area_vs_approval.png
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── shap_summary.png
    └── model_comparison.json
```

---

## 🚀 Option A — Run in Google Colab (recommended for this project)

Open a new Colab notebook and run each cell below in order.

### Cell 1 — Upload the project files
Either upload the whole folder as a zip and unzip it, or upload files
individually with the Colab file panel. If using a zip:

```python
from google.colab import files
uploaded = files.upload()  # select credilens_ai.zip

import zipfile
with zipfile.ZipFile("credilens_ai.zip", "r") as z:
    z.extractall(".")

%cd credilens_ai
```

### Cell 2 — Install dependencies

```python
!pip install -q -r requirements.txt
```

### Cell 3 — Generate the dataset

```python
!python generate_dataset.py
```

### Cell 4 — Train models, run SHAP, save all artifacts

```python
!python model_training.py
```

This will print accuracy/precision/recall/F1/ROC-AUC for all 4 models,
declare the best one (selected by ROC-AUC), and save:
- `models/trained_model.pkl`
- `models/preprocessor.pkl`
- `models/shap_explainer.pkl`
- `models/metadata.json`
- All charts in `assets/`

### Cell 5 — Install Node.js tunnel tool (one-time)

```python
!npm install -g localtunnel
```

### Cell 6 — Launch the Streamlit app

```python
!streamlit run app.py --server.port 8501 --server.headless true &>/content/logs.txt &
```

### Cell 7 — Expose it publicly with a tunnel

```python
import urllib.request
print("Tunnel password:", urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip())

!npx localtunnel --port 8501
```

Click the `https://xxxx.loca.lt` link that's printed, then paste the
**Tunnel password** shown above when prompted. CrediLens AI will load in
your browser.

> 💡 Alternative tunnels: if `localtunnel` is flaky, you can instead use
> `pyngrok` (`pip install pyngrok`, requires a free ngrok auth token).

---

## 💻 Option B — Run locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python generate_dataset.py

# 4. Train models + run SHAP + save artifacts
python model_training.py

# 5. Launch the dashboard
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🧠 What's Under the Hood

### Machine Learning
- **4 models trained & compared:** Logistic Regression, Decision Tree,
  Random Forest, XGBoost — selected automatically by highest ROC-AUC.
- **Feature engineering:** Total Income, Loan-to-Income Ratio, estimated
  EMI, and Disposable Balance Income — the same signals real underwriters use.
- **Missing-value handling:** mode imputation for categoricals, median for
  numerics — robust to outlier incomes.
- **Explainability:** SHAP `TreeExplainer` for tree models (or
  `LinearExplainer` for Logistic Regression), surfaced both as a global
  summary plot and as a per-application top-5 factor breakdown.

### Web App (Streamlit)
- **🏠 Dashboard** — KPIs, approval distribution, model comparison, risk heatmap.
- **📋 Loan Application** — full banking-style intake form.
- **📊 Prediction Result** — Approved/Rejected badge, animated probability
  gauge, SHAP-based "why" explanation, downloadable TXT/PDF report.
- **📈 Analytics** — feature importance, SHAP summary, approval trends,
  risk segmentation scatter, correlation heatmap.
- **📁 Bulk Prediction** — upload a CSV of many applications, get instant
  batch decisions + downloadable results.

---

## 🎨 Design System
- Background: deep navy `#0A1F44` gradient
- Accent: gold `#D4AF37`
- Glassmorphism cards with blur + hover lift
- Typography: Poppins (headings) / Inter (body)
- Status colors: green (approved), red (rejected), gold (borderline risk)

---

## ⚠️ Notes
- The dataset shipped here is **synthetically generated** with realistic,
  statistically grounded relationships (e.g. credit history dominates the
  decision, exactly as in real underwriting). To use a real dataset (such
  as the Kaggle "Loan Prediction" dataset), just replace
  `data/loan_data.csv` with your file using the same column schema, and
  re-run `model_training.py`.
- This is a **demonstration / portfolio system**. Predictions are not
  financial advice and should not be used for real lending decisions
  without proper regulatory validation, bias auditing, and compliance review.
