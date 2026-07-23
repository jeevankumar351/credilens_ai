"""
CrediLens AI - Preprocessing Module
====================================
Handles all data cleaning, encoding, and feature engineering for the
loan approval model. This module is imported by both model_training.py
and app.py to guarantee the exact same transformation is applied at
training time and at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
ENGINEERED_COLS = [
    "TotalIncome", "LoanIncomeRatio", "EMI", "BalanceIncome",
]
FEATURE_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS + ENGINEERED_COLS

TARGET_COL = "Loan_Status"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain-specific engineered features used by real loan underwriters."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Avoid divide-by-zero; LoanAmount is stored in thousands in this schema
    safe_loan = df["LoanAmount"].replace(0, np.nan)
    df["LoanIncomeRatio"] = (safe_loan * 1000) / df["TotalIncome"].replace(0, np.nan)
    df["LoanIncomeRatio"] = df["LoanIncomeRatio"].fillna(df["LoanIncomeRatio"].median())

    # Approximate monthly EMI (Equated Monthly Installment) — flat estimate
    safe_term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = (df["LoanAmount"] * 1000) / safe_term
    df["EMI"] = df["EMI"].fillna(df["EMI"].median())

    # Income remaining after EMI is paid — a key underwriting signal
    df["BalanceIncome"] = (df["TotalIncome"] / 12) - df["EMI"]

    return df


def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Handles missing values using sensible, banking-appropriate imputation:
      - Categorical -> mode
      - Numeric -> median (robust to income outliers)
      - Credit_History -> mode (it is effectively categorical: 0/1)
    """
    df = df.copy()

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    if "Credit_History" in df.columns and df["Credit_History"].isnull().any():
        df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    if "Loan_Amount_Term" in df.columns and df["Loan_Amount_Term"].isnull().any():
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    if "LoanAmount" in df.columns and df["LoanAmount"].isnull().any():
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    return df


class LoanPreprocessor:
    """
    Stateful preprocessor: fit on training data, then apply identical
    transformations (encoding + scaling) to any new data at inference time.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame):
        df = clean_data(df, is_training=True)
        df = _engineer_features(df)

        # Encode target
        y = df[TARGET_COL].map({"Y": 1, "N": 0}).values

        X = df[self.feature_columns].copy()

        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

        self.is_fitted = True
        return X_scaled, y

    def transform(self, df: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        df = clean_data(df, is_training=False)
        df = _engineer_features(df)

        X = df[self.feature_columns].copy()

        for col in CATEGORICAL_COLS:
            le = self.label_encoders[col]
            X[col] = X[col].astype(str).apply(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            X[col] = le.transform(X[col])

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        return X_scaled

    def save(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/preprocessor.pkl"):
        return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv("data/loan_data.csv")
    pre = LoanPreprocessor()
    X, y = pre.fit_transform(df)
    print("✅ Preprocessing complete")
    print("Feature matrix shape:", X.shape)
    print("Target distribution:", np.bincount(y))
    print("\nFeature columns:", list(X.columns))
    pre.save()
    print("✅ Preprocessor saved to models/preprocessor.pkl")
