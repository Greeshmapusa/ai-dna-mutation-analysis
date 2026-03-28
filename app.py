import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from fpdf import FPDF

# -------------------------------
# Load AI Models
# -------------------------------
model = joblib.load("cancer_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------
# Gene Functional Explanations
# -------------------------------
gene_function = {
    "BRCA1": "DNA repair and breast/ovarian cancer risk",
    "BRCA2": "DNA repair and breast/ovarian cancer risk",
    "TP53": "Tumor suppressor, prevents abnormal cell growth",
    "ATM": "DNA repair and genome stability",
    "PIK3CA": "Promotes cell growth and proliferation",
    "KRAS": "Drives uncontrolled cell division"
}
default_function = "Acts as a guardian gene controlling cell growth or preventing DNA errors."

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config("DNA Mutation AI Risk Predictor", "🧬", layout="wide")
st.title("🧬 AI-Driven DNA Mutation Analysis for Early Cancer Risk Prediction")
st.caption("Multi-Class Cancer Risk Prediction + Explainable AI")

# -------------------------------
# INPUT METHOD SELECTION (NEW)
# -------------------------------
input_type = st.selectbox(
    "Select Input Method",
    ["CSV Upload", "Paste Genetic Report", "Use Sample Data"]
)

mutation_text = ""

# -------------------------------
# CSV INPUT
# -------------------------------
if input_type == "CSV Upload":
    uploaded = st.file_uploader("Upload DNA Mutation CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "Gene" in df.columns and "Variation" in df.columns:
            mutation_text = " ".join(df["Gene"] + " " + df["Variation"])
            st.subheader("📄 Uploaded DNA Data")
            st.dataframe(df)
        else:
            st.warning("CSV must contain 'Gene' and 'Variation' columns!")

# -------------------------------
# TEXT INPUT (IMPROVED)
# -------------------------------
elif input_type == "Paste Genetic Report":
    report = st.text_area("Paste Genetic Test Report")

    if report:
        lines = report.strip().split("\n")
        parsed = []

        for line in lines:
            if "|" in line:
                try:
                    parts = line.split("|")
                    gene = parts[0].split(":")[1].strip().upper()
                    mutation = parts[1].split(":")[1].strip().upper()
                    parsed.append(gene + " " + mutation)
                except:
                    continue

        if parsed:
            mutation_text = " ".join(parsed)
        else:
            st.warning("Invalid format! Use: Gene: EGFR | Mutation: L858R")

# -------------------------------
# SAMPLE DATA (NEW)
# -------------------------------
elif input_type == "Use Sample Data":
    sample_data = """Gene: EGFR | Mutation: L858R
Gene: TP53 | Mutation: R175H
Gene: KRAS | Mutation: G12D"""

    st.text_area("Sample Input", sample_data)

    parsed = []
    for line in sample_data.split("\n"):
        parts = line.split("|")
        gene = parts[0].split(":")[1].strip().upper()
        mutation = parts[1].split(":")[1].strip().upper()
        parsed.append(gene + " " + mutation)

    mutation_text = " ".join(parsed)

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.button("🔬 Analyze DNA Mutations"):

    if not mutation_text:
        st.warning("Please provide valid input!")
    else:

        # Transform input
        X = vectorizer.transform([mutation_text])
        prob = model.predict_proba(X)[0]
        pred_class = np.argmax(prob)

        # -------------------------------
        # Cancer Mapping
        # -------------------------------
        num_classes = len(model.classes_)
        cancer_mapping = {i: f"Cancer Class {i}" for i in range(num_classes)}

        predefined_mapping = {
            0: "No Significant Risk",
            1: "Breast Cancer Risk",
            2: "Lung Cancer Risk",
            3: "Colon Cancer Risk",
            4: "Prostate Cancer Risk",
            5: "Ovarian Cancer Risk",
            6: "Liver Cancer Risk",
            7: "Pancreatic Cancer Risk",
            8: "Stomach Cancer Risk"
        }

        for k, v in predefined_mapping.items():
            if k < num_classes:
                cancer_mapping[k] = v

        pred_label = cancer_mapping.get(pred_class)

        # -------------------------------
        # Risk Level
        # -------------------------------
        if prob[pred_class] >= 0.7:
            risk_level = "HIGH"
        elif prob[pred_class] >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.subheader("📄 AI DNA Mutation Risk Report")
        st.write("**Patient ID:** P1024")
        st.write("**Predicted Cancer Type:**", pred_label)
        st.write("**Risk Level:**", risk_level)

        # -------------------------------
        # Probability Table
        # -------------------------------
        st.subheader("🎯 Probability Distribution")
        prob_df = pd.DataFrame({
            "Cancer Type": [cancer_mapping[i] for i in range(len(prob))],
            "Probability (%)": [round(p * 100, 2) for p in prob]
        })
        st.dataframe(prob_df)

        # -------------------------------
        # Explainable AI
        # -------------------------------
        st.subheader("🧬 Top Genetic Contributors")

        X_array = X.toarray()[0]
        features = vectorizer.get_feature_names_out()
        top_idx = X_array.argsort()[-6:][::-1]

        for i in top_idx:
            feature_name = features[i]
            parts = feature_name.split()

            gene_symbol = parts[0].upper()
            mutation_code = " ".join(parts[1:]) if len(parts) > 1 else ""

            contribution = round(X_array[i] * 100, 2)
            function_desc = gene_function.get(gene_symbol, default_function)

            st.write(f"**{gene_symbol} ({mutation_code})** → {contribution}%")
            st.write(f"Function: {function_desc}")
            st.progress(min(contribution / 100, 1))

        # -------------------------------
        # PDF GENERATION
        # -------------------------------
        if st.button("📄 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, "AI DNA Mutation Risk Report", ln=True)
            pdf.cell(200, 10, f"Patient ID: P1024", ln=True)
            pdf.cell(200, 10, f"Cancer Type: {pred_label}", ln=True)
            pdf.cell(200, 10, f"Risk Level: {risk_level}", ln=True)

            pdf.ln(5)
            pdf.cell(200, 10, "Probability Distribution:", ln=True)

            for i, p in enumerate(prob):
                pdf.cell(200, 10, f"{cancer_mapping[i]}: {round(p*100,2)}%", ln=True)

            pdf.output("DNA_Risk_Report.pdf")
            st.success("Report Generated Successfully!")
