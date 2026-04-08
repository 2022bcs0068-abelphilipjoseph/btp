# Implementation Plan - Agentic BERT Bias Dashboard

This plan outlines the creation of a Streamlit dashboard to analyze gender bias in resume screening using BERT models. It incorporates an "Agentic Auditor" to provide natural language explanations of bias detected by the models.

## Proposed Changes

### Dashboard Component
#### [MODIFY] [dashboard.py](file:///Users/macbookpro/Desktop/BTP/explainability/dashboard.py)
A Streamlit application that provides:
- **Joint Analysis**: New input field for **Job Description (JD)**.
- **Scoring**: BERT scores based on JD-Resume concatenation (`JD [SEP] Resume`).
- **Bias Flip Examples**: Pre-set buttons for "Male (Less Qualified)" and "Female (More Qualified)" for demonstration.
- **Explainability Suite**: LRP, SHAP, LIME, and Attention visualizations.
- **Agentic Auditor**: Summary section using Qwen to analyze the "Bias Flip" phenomenon.

### Explanation Logic
#### [MODIFY] [main.py](file:///Users/macbookpro/Desktop/BTP/explainability/main.py)
Refactor existing explanation functions to be more modular so they can be easily imported and used by the Streamlit dashboard. Specifically, ensure LRP and SHAP functions return figures or data structures compatible with Streamlit.

## Verification Plan

### Manual Verification
1.  **Environment Setup**: Install `streamlit` and `seaborn` if not already present.
2.  **Launch Dashboard**: Run `streamlit run explainability/dashboard.py` in the terminal.
3.  **Upload Resumes**: 
    - Prepare two similar resumes: one with a male name (e.g., "John Smith") and one with a female name (e.g., "Jane Smith").
    - Upload both to the dashboard.
4.  **Bias Analysis**:
    - Observe the scoring difference in the "Before" model.
    - Observe if the difference is reduced in the "After" model.
5.  **Explanations**:
    - Click into "Explainability Detail" to view heatmaps.
    - Verify that "Agentic Auditor" provides a coherent summary of the bias reduction.
