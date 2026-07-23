"""
CrediLens AI - Dataset Generator
=================================
Generates a realistic, structured loan-approval dataset that mirrors the
classic "Loan Prediction" schema used across the banking/fintech industry.

If you already have a real dataset (e.g. the Kaggle "Loan Prediction"
dataset), simply place it at data/loan_data.csv with the same column
names and SKIP this script.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 1500

np.random.seed(RANDOM_SEED)


def generate_loan_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Generates a synthetic but statistically realistic loan dataset."""

    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.78, 0.22])
    married = np.random.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    dependents = np.random.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.58, 0.17, 0.17, 0.08])
    education = np.random.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = np.random.choice(["Yes", "No"], size=n_samples, p=[0.14, 0.86])
    property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.38, 0.38, 0.24])

    # Income correlated with education
    base_income = np.where(
        education == "Graduate",
        np.random.gamma(shape=5.0, scale=1200, size=n_samples) + 2500,
        np.random.gamma(shape=4.0, scale=900, size=n_samples) + 1500,
    )
    applicant_income = np.round(base_income, -1).astype(int)

    # Coapplicant income (0 if not married, otherwise some chance of income)
    coapplicant_income = np.where(
        married == "Yes",
        np.where(np.random.rand(n_samples) < 0.55,
                 np.round(np.random.gamma(shape=3.0, scale=700, size=n_samples), -1), 0),
        0,
    ).astype(int)

    # Loan amount correlated with combined income
    combined_income = applicant_income + coapplicant_income
    loan_amount = np.round(
        (combined_income * np.random.uniform(0.08, 0.22, size=n_samples)), -1
    ).astype(int)
    loan_amount = np.clip(loan_amount, 10, 700)

    loan_amount_term = np.random.choice(
        [360, 180, 240, 120, 60, 300, 84, 36, 12],
        size=n_samples,
        p=[0.72, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02],
    )

    # Credit history (1 = good history, 0 = bad/no history)
    credit_history = np.random.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])

    df = pd.DataFrame({
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    })

    # ---- Generate target variable (Loan_Status) using a realistic latent score ----
    score = np.full(n_samples, -1.1)  # base offset to balance approval rate
    score += np.where(df["Credit_History"] == 1.0, 2.3, -3.0)          # dominant factor
    score += np.where(df["Education"] == "Graduate", 0.35, -0.25)
    score += np.where(df["Married"] == "Yes", 0.2, 0.0)
    score += np.where(df["Self_Employed"] == "Yes", -0.15, 0.05)
    score += np.where(df["Property_Area"] == "Semiurban", 0.45,
                       np.where(df["Property_Area"] == "Urban", 0.05, -0.25))

    income_to_loan = combined_income / (df["LoanAmount"] * 1000 + 1)
    score += np.clip(income_to_loan * 6, -1.8, 1.8)

    dependents_penalty = df["Dependents"].map({"0": 0.1, "1": 0.0, "2": -0.15, "3+": -0.35})
    score += dependents_penalty

    term_factor = np.where(df["Loan_Amount_Term"] >= 300, 0.1, -0.1)
    score += term_factor

    # Add noise for realism
    score += np.random.normal(0, 1.0, size=n_samples)

    prob_approved = 1 / (1 + np.exp(-score))
    loan_status = np.where(prob_approved > 0.5, "Y", "N")

    df["Loan_Status"] = loan_status

    # ---- Inject realistic missing values (banking data is never clean) ----
    for col, frac in [
        ("Gender", 0.02), ("Married", 0.005), ("Dependents", 0.025),
        ("Self_Employed", 0.05), ("LoanAmount", 0.035),
        ("Loan_Amount_Term", 0.02), ("Credit_History", 0.08),
    ]:
        n_missing = int(n_samples * frac)
        idx = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan

    # Loan_ID column (banking systems always have an application ID)
    df.insert(0, "Loan_ID", [f"LN{100000 + i}" for i in range(n_samples)])

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_loan_dataset()
    out_path = os.path.join("data", "loan_data.csv")
    dataset.to_csv(out_path, index=False)
    print(f"✅ Dataset generated: {out_path}")
    print(f"Shape: {dataset.shape}")
    print(f"\nApproval distribution:\n{dataset['Loan_Status'].value_counts(normalize=True)}")
    print(f"\nMissing values:\n{dataset.isnull().sum()}")
