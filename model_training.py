"""
CrediLens AI - Model Training Pipeline
========================================
Trains and compares 4 models for loan approval prediction:
    1. Logistic Regression (baseline)
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. XGBoost Classifier (production model)

Selects the best model based on ROC-AUC, generates SHAP explainability
artifacts, and saves everything needed for the Streamlit app.

Run:
    python model_training.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    # Fallback so the pipeline still runs end-to-end in restricted sandboxes.
    # In Colab / any normal environment, `pip install xgboost` makes this
    # branch unnecessary — XGBClassifier is used as specified.
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    XGBOOST_AVAILABLE = False
    print("⚠️  xgboost not found — falling back to GradientBoostingClassifier. "
          "Run `pip install xgboost` for the true production model.")

from preprocessing import LoanPreprocessor

RANDOM_SEED = 42
MODELS_DIR = "models"
ASSETS_DIR = "assets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

GOLD = "#D4AF37"
NAVY = "#0A1F44"
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def load_and_split_data(path: str = "data/loan_data.csv"):
    df = pd.read_csv(path)
    preprocessor = LoanPreprocessor()
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor, df


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False,
        ) if XGBOOST_AVAILABLE else XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=RANDOM_SEED,
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    models = get_candidate_models()
    results = {}
    fitted_models = {}

    print("\n" + "=" * 60)
    print("TRAINING & COMPARING MODELS")
    print("=" * 60)

    for name, model in models.items():
        if name == "XGBoost" and XGBOOST_AVAILABLE:
            # xgboost expects no use_label_encoder kwarg in newer versions on .fit
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model

        print(f"\n📊 {name}")
        print(f"   Accuracy : {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall   : {metrics['recall']:.4f}")
        print(f"   F1-Score : {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC  : {metrics['roc_auc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
    print("=" * 60)

    return fitted_models, results, best_name


def save_comparison_table(results: dict, path: str = f"{ASSETS_DIR}/model_comparison.json"):
    table = {
        name: {
            "accuracy": round(float(m["accuracy"]), 4),
            "precision": round(float(m["precision"]), 4),
            "recall": round(float(m["recall"]), 4),
            "f1_score": round(float(m["f1_score"]), 4),
            "roc_auc": round(float(m["roc_auc"]), 4),
        }
        for name, m in results.items()
    }
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"✅ Model comparison table saved to {path}")
    return table


# ----------------------------- VISUALIZATIONS -----------------------------

def plot_roc_curves(results: dict, y_test, path: str = f"{ASSETS_DIR}/roc_curves.png"):
    plt.figure(figsize=(7, 6))
    colors = [GOLD, "#4A90D9", "#5CB85C", "#D9534F"]
    for (name, m), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})", color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ ROC curves saved to {path}")


def plot_confusion_matrix(y_test, y_pred, model_name: str,
                           path: str = f"{ASSETS_DIR}/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"], cbar=False, linewidths=1, linecolor="white")
    plt.title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")


def plot_feature_importance(model, feature_names, model_name: str,
                             path: str = f"{ASSETS_DIR}/feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    order = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color=GOLD, edgecolor=NAVY)
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Feature importance saved to {path}")
    return dict(zip(sorted_features.tolist(), sorted_importances.tolist()))


def plot_eda_charts(df: pd.DataFrame, path_prefix: str = ASSETS_DIR):
    # 1. Loan approval distribution (pie chart)
    plt.figure(figsize=(5, 5))
    counts = df["Loan_Status"].value_counts()
    plt.pie(counts, labels=["Approved", "Rejected"] if counts.index[0] == "Y" else ["Rejected", "Approved"],
            autopct="%1.1f%%", colors=[GOLD, NAVY], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    plt.title("Loan Approval Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/approval_distribution.png", dpi=150)
    plt.close()

    # 2. Income vs Approval rate
    plt.figure(figsize=(7, 5))
    df_plot = df.copy()
    df_plot["IncomeBracket"] = pd.qcut(df_plot["ApplicantIncome"], 5, duplicates="drop")
    approval_by_income = df_plot.groupby("IncomeBracket", observed=True)["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    )
    approval_by_income.plot(kind="bar", color=GOLD, edgecolor=NAVY)
    plt.ylabel("Approval Rate (%)")
    plt.xlabel("Applicant Income Bracket")
    plt.title("Income vs Approval Rate", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/income_vs_approval.png", dpi=150)
    plt.close()

    # 3. Credit History vs Approval
    plt.figure(figsize=(6, 5))
    ch_approval = df.groupby("Credit_History")["Loan_Status"].apply(lambda x: (x == "Y").mean() * 100)
    ch_approval.index = ["Bad/No History" if i == 0 else "Good History" for i in ch_approval.index]
    ch_approval.plot(kind="bar", color=[NAVY, GOLD], edgecolor="white")
    plt.ylabel("Approval Rate (%)")
    plt.title("Credit History vs Approval Rate", fontweight="bold")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/credit_history_vs_approval.png", dpi=150)
    plt.close()

    # 4. Property Area risk comparison
    plt.figure(figsize=(6, 5))
    area_approval = df.groupby("Property_Area")["Loan_Status"].apply(lambda x: (x == "Y").mean() * 100)
    area_approval.plot(kind="bar", color=GOLD, edgecolor=NAVY)
    plt.ylabel("Approval Rate (%)")
    plt.title("Property Area vs Approval Rate", fontweight="bold")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}/property_area_vs_approval.png", dpi=150)
    plt.close()

    print("✅ EDA charts saved to assets/")


def run_shap_analysis(model, X_train, X_test, feature_names, model_name: str):
    """Generates SHAP summary plot and saves a SHAP explainer for live use in the app."""
    try:
        import shap
    except ImportError:
        print("⚠️  shap not installed — skipping SHAP analysis. Run `pip install shap`.")
        return None

    print("\n🔍 Running SHAP explainability analysis...")

    background = X_train.sample(min(100, len(X_train)), random_state=RANDOM_SEED)

    if model_name in ("Random Forest", "Decision Tree", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, background)

    sample = X_test.sample(min(200, len(X_test)), random_state=RANDOM_SEED)
    shap_values = explainer.shap_values(sample)

    # Handle binary classifiers that return a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    plt.figure()
    shap.summary_plot(shap_values_plot, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP summary plot saved to {ASSETS_DIR}/shap_summary.png")

    # Save the explainer itself for per-prediction explanations in the app
    joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")
    print(f"✅ SHAP explainer saved to {MODELS_DIR}/shap_explainer.pkl")

    return explainer


def main():
    print("🚀 CrediLens AI — Model Training Pipeline")
    print("=" * 60)

    X_train, X_test, y_train, y_test, preprocessor, raw_df = load_and_split_data()
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    fitted_models, results, best_name = train_and_compare_models(X_train, X_test, y_train, y_test)
    comparison_table = save_comparison_table(results)

    best_model = fitted_models[best_name]
    best_metrics = results[best_name]

    # Visualizations
    plot_roc_curves(results, y_test)
    plot_confusion_matrix(y_test, best_metrics["y_pred"], best_name)
    importance_dict = plot_feature_importance(best_model, X_train.columns.tolist(), best_name)
    plot_eda_charts(raw_df)

    # SHAP
    run_shap_analysis(best_model, X_train, X_test, X_train.columns.tolist(), best_name)

    # Save final artifacts
    joblib.dump(best_model, f"{MODELS_DIR}/trained_model.pkl")
    preprocessor.save(f"{MODELS_DIR}/preprocessor.pkl")

    metadata = {
        "best_model_name": best_name,
        "xgboost_used": XGBOOST_AVAILABLE,
        "metrics": {k: round(float(v), 4) for k, v in best_metrics.items()
                    if k not in ("y_pred", "y_proba")},
        "feature_columns": X_train.columns.tolist(),
        "feature_importance": importance_dict,
        "comparison_table": comparison_table,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print(f"   Production model : {best_name}")
    print(f"   Saved to         : {MODELS_DIR}/trained_model.pkl")
    print(f"   Preprocessor     : {MODELS_DIR}/preprocessor.pkl")
    print(f"   Metadata         : {MODELS_DIR}/metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
