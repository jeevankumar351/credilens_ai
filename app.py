"""
CrediLens AI — Intelligent Loan Approval System
=================================================
Bank-grade Streamlit dashboard for AI-powered loan decisioning.

Run:
    streamlit run app.py
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from preprocessing import LoanPreprocessor

try:
    from pdf_report import generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CrediLens AI | Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"
ASSETS_DIR = "assets"
DATA_PATH = "data/loan_data.csv"

NAVY = "#0A1F44"
NAVY_LIGHT = "#10295C"
GOLD = "#D4AF37"
GOLD_LIGHT = "#E8C766"
GREEN = "#3DDC97"
RED = "#FF5C5C"

# --------------------------------------------------------------------------
# GLOBAL CSS — Bank-grade glassmorphism dark theme
# --------------------------------------------------------------------------
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: linear-gradient(160deg, {NAVY} 0%, #081530 55%, #050d1f 100%);
        color: #EAF0FF;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #081530 0%, {NAVY} 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.25);
    }}

    h1, h2, h3, h4 {{
        font-family: 'Poppins', sans-serif !important;
        color: #F5F8FF !important;
        font-weight: 700 !important;
    }}

    p, span, label, div {{
        color: #D9E2F5;
    }}

    /* Glass card */
    .glass-card {{
        background: rgba(255, 255, 255, 0.045);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(212, 175, 55, 0.22);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 1rem;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(212, 175, 55, 0.18);
    }}

    .metric-label {{
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {GOLD_LIGHT};
        font-weight: 600;
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: #9FB3D9;
        margin-top: 0.2rem;
    }}

    .brand-title {{
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        background: linear-gradient(90deg, {GOLD_LIGHT}, {GOLD});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }}
    .brand-sub {{
        color: #9FB3D9;
        font-size: 0.95rem;
        margin-top: -6px;
        letter-spacing: 0.03em;
    }}

    .badge-approved {{
        background: linear-gradient(135deg, {GREEN}, #2BB67E);
        color: #06251A;
        padding: 0.6rem 1.3rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        box-shadow: 0 4px 18px rgba(61, 220, 151, 0.35);
    }}
    .badge-rejected {{
        background: linear-gradient(135deg, {RED}, #C73E3E);
        color: #2A0606;
        padding: 0.6rem 1.3rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        box-shadow: 0 4px 18px rgba(255, 92, 92, 0.35);
    }}

    .explain-box {{
        background: rgba(212, 175, 55, 0.07);
        border-left: 3px solid {GOLD};
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin: 0.45rem 0;
        font-size: 0.92rem;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {GOLD}, #B8932E);
        color: {NAVY};
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.3);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(212, 175, 55, 0.45);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        color: #B8C6E8;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(212, 175, 55, 0.18) !important;
        color: {GOLD_LIGHT} !important;
    }}

    hr {{
        border-color: rgba(212, 175, 55, 0.2);
    }}

    [data-testid="stMetricValue"] {{
        color: {GOLD_LIGHT};
    }}

    .footer-note {{
        text-align: center;
        color: #6F84B5;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(212,175,55,0.15);
    }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/trained_model.pkl")
    preprocessor = LoanPreprocessor.load(f"{MODELS_DIR}/preprocessor.pkl")
    with open(f"{MODELS_DIR}/metadata.json") as f:
        metadata = json.load(f)

    shap_explainer = None
    shap_path = f"{MODELS_DIR}/shap_explainer.pkl"
    if os.path.exists(shap_path):
        try:
            shap_explainer = joblib.load(shap_path)
        except Exception:
            shap_explainer = None

    return model, preprocessor, metadata, shap_explainer


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def artifacts_exist():
    return (
        os.path.exists(f"{MODELS_DIR}/trained_model.pkl")
        and os.path.exists(f"{MODELS_DIR}/preprocessor.pkl")
        and os.path.exists(f"{MODELS_DIR}/metadata.json")
        and os.path.exists(DATA_PATH)
    )


if not artifacts_exist():
    st.error(
        "⚠️ Model artifacts not found. Please run the training pipeline first:\n\n"
        "```bash\npython generate_dataset.py\npython model_training.py\n```"
    )
    st.stop()

model, preprocessor, metadata, shap_explainer = load_artifacts()
raw_df = load_dataset()

FEATURE_COLUMNS = metadata["feature_columns"]
BEST_MODEL_NAME = metadata["best_model_name"]


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def build_input_dataframe(form: dict) -> pd.DataFrame:
    row = {
        "Gender": form["gender"],
        "Married": form["married"],
        "Dependents": form["dependents"],
        "Education": form["education"],
        "Self_Employed": form["self_employed"],
        "ApplicantIncome": form["applicant_income"],
        "CoapplicantIncome": form["coapplicant_income"],
        "LoanAmount": form["loan_amount"],
        "Loan_Amount_Term": form["loan_term"],
        "Credit_History": 1.0 if form["credit_history"] else 0.0,
        "Property_Area": form["property_area"],
    }
    return pd.DataFrame([row])


def predict_loan(input_df: pd.DataFrame):
    X = preprocessor.transform(input_df)
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return pred, proba, X


def risk_gauge(probability_approved: float):
    """Plotly gauge meter showing approval probability (0-100%)."""
    pct = probability_approved * 100
    if pct >= 70:
        bar_color = GREEN
    elif pct >= 45:
        bar_color = GOLD
    else:
        bar_color = RED

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": "#FFFFFF", "family": "Poppins"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9FB3D9", "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "rgba(255, 92, 92, 0.18)"},
                {"range": [45, 70], "color": "rgba(212, 175, 55, 0.18)"},
                {"range": [70, 100], "color": "rgba(61, 220, 151, 0.18)"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D9E2F5"},
    )
    return fig


def get_shap_explanation(X_row: pd.DataFrame, top_n: int = 5):
    """Returns a list of (feature, shap_value, direction) for a single prediction."""
    if shap_explainer is None:
        return None
    try:
        shap_vals = shap_explainer.shap_values(X_row)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.array(shap_vals).flatten()
        feature_names = X_row.columns.tolist()

        pairs = list(zip(feature_names, shap_vals))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_n]
    except Exception:
        return None


def fallback_explanation(input_row: dict, top_n: int = 5):
    """Rule-based explanation fallback if SHAP isn't available, using feature importances."""
    importance = metadata.get("feature_importance", {})
    if not importance:
        return []

    explanations = []
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for feat, imp in sorted_feats:
        explanations.append((feat, imp, None))
    return explanations


FRIENDLY_NAMES = {
    "Credit_History": "Credit History",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Co-applicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Term",
    "TotalIncome": "Total Household Income",
    "LoanIncomeRatio": "Loan-to-Income Ratio",
    "EMI": "Estimated Monthly Installment",
    "BalanceIncome": "Disposable Income After EMI",
    "Property_Area": "Property Area",
    "Self_Employed": "Self-Employment Status",
    "Dependents": "Number of Dependents",
    "Education": "Education Level",
    "Married": "Marital Status",
    "Gender": "Gender",
}


def render_explanation_cards(pairs, prediction_label: str):
    if not pairs:
        st.info("Explanation data unavailable for this model.")
        return
    for feat, val, _ in pairs:
        name = FRIENDLY_NAMES.get(feat, feat)
        if isinstance(val, (int, float, np.floating)):
            if prediction_label == "Approved":
                direction = "supported approval ✅" if val > 0 or _ is None else "worked against approval ⚠️"
            else:
                direction = "contributed to rejection ⚠️" if val > 0 or _ is None else "worked in the applicant's favor ✅"
            st.markdown(
                f"""<div class="explain-box"><b>{name}</b> — {direction}
                <span style="color:#9FB3D9; font-size:0.82rem;"> (impact score: {abs(val):.3f})</span></div>""",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""<div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
        <div style="font-size:2.4rem;">🏦</div>
        <div class="brand-title">CrediLens AI</div>
        <div class="brand-sub">Loan Intelligence Engine</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📋 Loan Application", "📊 Prediction Result", "📈 Analytics", "📁 Bulk Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""<div style="font-size:0.78rem; color:#9FB3D9;">
        <b>Production Model:</b> {BEST_MODEL_NAME}<br>
        <b>ROC-AUC:</b> {metadata['metrics']['roc_auc']:.3f}<br>
        <b>Accuracy:</b> {metadata['metrics']['accuracy']:.1%}<br>
        </div>""",
        unsafe_allow_html=True,
    )
    if not metadata.get("xgboost_used", True):
        st.caption("⚠️ Trained with GradientBoosting fallback (xgboost not installed in this env).")


# --------------------------------------------------------------------------
# PAGE: DASHBOARD
# --------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown('<div class="brand-title" style="font-size:2.3rem;">Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Real-time portfolio overview &amp; model health</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    total_apps = len(raw_df)
    approved = (raw_df["Loan_Status"] == "Y").sum()
    approval_rate = approved / total_apps * 100
    avg_loan = raw_df["LoanAmount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in zip(
        [c1, c2, c3, c4],
        ["Total Applications", "Approval Rate", "Avg. Loan Amount", "Model ROC-AUC"],
        [f"{total_apps:,}", f"{approval_rate:.1f}%", f"₹{avg_loan*1000:,.0f}", f"{metadata['metrics']['roc_auc']:.3f}"],
        ["Historical dataset", f"{approved:,} approved", "Across portfolio", f"{BEST_MODEL_NAME}"],
    ):
        with col:
            st.markdown(
                f"""<div class="glass-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Loan Status Distribution")
        status_counts = raw_df["Loan_Status"].map({"Y": "Approved", "N": "Rejected"}).value_counts()
        fig = px.pie(
            values=status_counts.values, names=status_counts.index,
            color=status_counts.index,
            color_discrete_map={"Approved": GREEN, "Rejected": RED},
            hole=0.55,
        )
        fig.update_traces(textinfo="percent+label", textfont_color="white")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#D9E2F5", showlegend=False, height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Model Performance Comparison")
        comp = metadata["comparison_table"]
        comp_df = pd.DataFrame(comp).T.reset_index().rename(columns={"index": "Model"})
        fig = px.bar(
            comp_df, x="Model", y="roc_auc", color="Model",
            color_discrete_sequence=[GOLD, "#4A90D9", "#5CB85C", "#D9534F"],
            text_auto=".3f",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#D9E2F5", showlegend=False, height=320,
            yaxis_title="ROC-AUC", xaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Risk Segmentation — Credit History vs Property Area")
    pivot = raw_df.copy()
    pivot["Approved"] = (pivot["Loan_Status"] == "Y").astype(int)
    pivot_table = pivot.pivot_table(
        index="Property_Area", columns="Credit_History", values="Approved", aggfunc="mean"
    ) * 100
    fig = px.imshow(
        pivot_table, text_auto=".1f", color_continuous_scale=["#0A1F44", GOLD],
        labels=dict(color="Approval %"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#D9E2F5", height=320, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE: LOAN APPLICATION
# --------------------------------------------------------------------------
elif page == "📋 Loan Application":
    st.markdown('<div class="brand-title" style="font-size:2.3rem;">New Loan Application</div>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Enter applicant details for instant AI-powered underwriting</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("loan_application_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 👤 Applicant Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self-Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 💰 Financial Details")
            applicant_income = st.slider("Applicant Monthly Income (₹)", 1000, 50000, 5000, step=500)
            coapplicant_income = st.slider("Co-applicant Monthly Income (₹)", 0, 30000, 0, step=500)
            loan_amount = st.slider("Loan Amount (in ₹ thousands)", 10, 700, 150, step=10)
            loan_term = st.select_slider(
                "Loan Term (months)",
                options=[12, 36, 60, 84, 120, 180, 240, 300, 360],
                value=360,
            )
            credit_history = st.toggle("Good Credit History", value=True)
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Run AI Underwriting", width='stretch')

        if submitted:
            st.session_state["form_data"] = {
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "credit_history": credit_history,
                "property_area": property_area,
            }
            st.session_state["predicted"] = True
            st.success("✅ Application submitted! Go to the **Prediction Result** page to view the AI decision.")


# --------------------------------------------------------------------------
# PAGE: PREDICTION RESULT
# --------------------------------------------------------------------------
elif page == "📊 Prediction Result":
    st.markdown('<div class="brand-title" style="font-size:2.3rem;">AI Decision Report</div>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Model prediction, risk score &amp; explainability</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "form_data" not in st.session_state:
        st.warning("⚠️ No application submitted yet. Please fill out the **Loan Application** form first.")
    else:
        form = st.session_state["form_data"]
        input_df = build_input_dataframe(form)
        pred, proba, X_row = predict_loan(input_df)

        prob_approved = proba[1]
        prediction_label = "Approved" if pred == 1 else "Rejected"
        confidence = max(proba) * 100

        col_left, col_right = st.columns([1, 1.3])

        with col_left:
            st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown("##### Decision")
            badge_class = "badge-approved" if pred == 1 else "badge-rejected"
            icon = "✅" if pred == 1 else "❌"
            st.markdown(
                f'<div class="{badge_class}">{icon} {prediction_label.upper()}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"<br><div class='metric-sub'>Model Confidence: <b>{confidence:.1f}%</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>Engine: {BEST_MODEL_NAME}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>Evaluated: {datetime.now().strftime('%d %b %Y, %H:%M')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### Approval Probability Meter")
            st.plotly_chart(risk_gauge(prob_approved), width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 🧠 AI Explanation — Why this decision?")
            st.caption("Top factors influencing this prediction (SHAP-based reasoning where available)")

            pairs = get_shap_explanation(X_row)
            if pairs is None:
                pairs = fallback_explanation(form)
            render_explanation_cards(pairs, prediction_label)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### Application Summary")
            summary_df = pd.DataFrame({
                "Field": ["Gender", "Married", "Dependents", "Education", "Self-Employed",
                          "Applicant Income", "Co-applicant Income", "Loan Amount (₹k)",
                          "Loan Term (months)", "Credit History", "Property Area"],
                "Value": [form["gender"], form["married"], form["dependents"], form["education"],
                          form["self_employed"], f"₹{form['applicant_income']:,}",
                          f"₹{form['coapplicant_income']:,}", form["loan_amount"],
                          form["loan_term"], "Good" if form["credit_history"] else "Poor/None",
                          form["property_area"]],
            })
            st.dataframe(summary_df, hide_index=True, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        # Downloadable report
        st.markdown("<br>", unsafe_allow_html=True)
        report_text = f"""
CREDILENS AI — LOAN DECISION REPORT
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Engine: {BEST_MODEL_NAME}

DECISION: {prediction_label.upper()}
Approval Probability: {prob_approved*100:.2f}%
Model Confidence: {confidence:.2f}%

APPLICANT DETAILS
-----------------
Gender: {form['gender']}
Married: {form['married']}
Dependents: {form['dependents']}
Education: {form['education']}
Self-Employed: {form['self_employed']}
Applicant Income: Rs.{form['applicant_income']:,}
Co-applicant Income: Rs.{form['coapplicant_income']:,}
Loan Amount: Rs.{form['loan_amount']}k
Loan Term: {form['loan_term']} months
Credit History: {'Good' if form['credit_history'] else 'Poor/None'}
Property Area: {form['property_area']}

This is an AI-generated risk assessment for demonstration purposes
and does not constitute a final lending decision.
"""
        st.download_button(
            "⬇️ Download Decision Report (.txt)",
            data=report_text,
            file_name=f"CrediLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            width='stretch',
        )

        if PDF_AVAILABLE:
            try:
                pdf_bytes = generate_pdf_report(
                    form=form,
                    prediction_label=prediction_label,
                    prob_approved=prob_approved,
                    confidence=confidence,
                    model_name=BEST_MODEL_NAME,
                    explanation_pairs=pairs,
                )
                st.download_button(
                    "📄 Download Decision Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"CrediLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    width='stretch',
                )
            except Exception as e:
                st.caption(f"PDF generation unavailable: {e}")


# --------------------------------------------------------------------------
# PAGE: ANALYTICS
# --------------------------------------------------------------------------
elif page == "📈 Analytics":
    st.markdown('<div class="brand-title" style="font-size:2.3rem;">Portfolio Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Deep-dive into model behavior &amp; portfolio trends</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Importance", "📈 Approval Trends", "🎯 Risk Segments", "🔗 Correlations"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        importance = metadata.get("feature_importance", {})
        if importance:
            imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
            imp_df["Feature"] = imp_df["Feature"].map(lambda f: FRIENDLY_NAMES.get(f, f))
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#10295C", GOLD],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#D9E2F5", height=480, margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width='stretch')
        if os.path.exists(f"{ASSETS_DIR}/shap_summary.png"):
            st.markdown("##### SHAP Summary Plot")
            st.image(f"{ASSETS_DIR}/shap_summary.png", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### Approval Rate by Education")
            edu = raw_df.copy()
            edu["Approved"] = (edu["Loan_Status"] == "Y").astype(int) * 100
            edu_agg = edu.groupby("Education")["Approved"].mean().reset_index()
            fig = px.bar(edu_agg, x="Education", y="Approved", color="Education",
                         color_discrete_sequence=[GOLD, NAVY_LIGHT])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#D9E2F5", height=350, showlegend=False,
                               yaxis_title="Approval %")
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### Approval Rate by Dependents")
            dep = raw_df.copy()
            dep["Approved"] = (dep["Loan_Status"] == "Y").astype(int) * 100
            dep_agg = dep.groupby("Dependents")["Approved"].mean().reset_index()
            fig = px.bar(dep_agg, x="Dependents", y="Approved", color="Dependents",
                         color_discrete_sequence=px.colors.sequential.YlOrBr)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#D9E2F5", height=350, showlegend=False,
                               yaxis_title="Approval %")
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Risk Segmentation: Income vs Loan Amount")
        seg = raw_df.copy()
        seg["Status"] = seg["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})
        fig = px.scatter(
            seg, x="ApplicantIncome", y="LoanAmount", color="Status",
            color_discrete_map={"Approved": GREEN, "Rejected": RED},
            opacity=0.65, size_max=8,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#D9E2F5", height=420)
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Correlation Heatmap (Numeric Features)")
        numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                         "Loan_Amount_Term", "Credit_History"]
        corr = raw_df[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale=["#0A1F44", "#FFFFFF", GOLD],
            zmin=-1, zmax=1,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#D9E2F5", height=450)
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE: BULK PREDICTION
# --------------------------------------------------------------------------
elif page == "📁 Bulk Prediction":
    st.markdown('<div class="brand-title" style="font-size:2.3rem;">Bulk Loan Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Upload a CSV of applications for batch underwriting</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### Required CSV columns")
    st.code(
        "Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, "
        "CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area",
        language="text",
    )

    template = raw_df.drop(columns=["Loan_ID", "Loan_Status"]).head(3)
    st.download_button(
        "⬇️ Download CSV Template",
        data=template.to_csv(index=False),
        file_name="credilens_bulk_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload applications CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            required_cols = [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History", "Property_Area",
            ]
            missing = [c for c in required_cols if c not in bulk_df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
            else:
                X_bulk = preprocessor.transform(bulk_df)
                preds = model.predict(X_bulk)
                probas = model.predict_proba(X_bulk)[:, 1]

                bulk_df["Prediction"] = np.where(preds == 1, "Approved", "Rejected")
                bulk_df["Approval_Probability"] = (probas * 100).round(2)

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("##### Batch Results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Applications", len(bulk_df))
                c2.metric("Approved", int((bulk_df["Prediction"] == "Approved").sum()))
                c3.metric("Rejected", int((bulk_df["Prediction"] == "Rejected").sum()))

                st.dataframe(bulk_df, width='stretch', hide_index=True)

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=bulk_df.to_csv(index=False),
                    file_name=f"CrediLens_BulkResults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch',
                )
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown(
    """<div class="footer-note">
    CrediLens AI © 2026 — Intelligent Loan Approval System &nbsp;|&nbsp;
    Built with Streamlit, Scikit-learn, XGBoost &amp; SHAP &nbsp;|&nbsp;
    For demonstration &amp; portfolio purposes only
    </div>""",
    unsafe_allow_html=True,
)
