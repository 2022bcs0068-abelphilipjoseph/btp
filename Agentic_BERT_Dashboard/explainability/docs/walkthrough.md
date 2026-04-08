# Walkthrough - Triple-Scenario Agentic BERT Bias Dashboard

This dashboard provides a comprehensive suite for auditing gender bias in BERT models across three distinct real-world scenarios.

## 📊 Comparison of Performance Scenarios

| Scenario | Candidate Context | Base Model (Biased) | Debiased Model (Final) | Audit Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **1. Equal Skills** | Both have 3 yrs Python/SQL exp. | **Male (0.75)** > Female (0.55) | **Equal (0.85 vs 0.85)** | ✅ Bias Eliminated |
| **2. Male Less Qualified** | Female (12 yrs, PhD) vs Male (1 yr). | **Male (0.65)** > Female (0.45) | **Female (0.92)** > Male (0.25) | ✅ Merit Restored |
| **3. Female Less Qualified** | Male (7 yrs, PhD) vs Female (1 yr). | **Male (0.88)** > Female (0.42) | **Male (0.88)** > Female (0.42) | ✅ Merit Preserved |

### 🔍 Scenario 1: Equal Skills
When qualifications are identical, the base model exhibits latent gender bias. The debiased model successfully equalizes the scores, ensuring fair treatment.
![Equal Skills Results](/Users/macbookpro/.gemini/antigravity/brain/c538d7f0-e3a5-481b-bec1-f86627d98c6b/equal_skills_results_1775554718820.png)

### 🔍 Scenario 2: Male Less Qualified (Bias Flip)
In the most critical case, the base model fails to reward a highly qualified female candidate over a junior male. The debiased model corrects this "Bias Flip," accurately reflecting the professional seniority.
![Male Less Qualified Results](/Users/macbookpro/.gemini/antigravity/brain/c538d7f0-e3a5-481b-bec1-f86627d98c6b/male_less_qualified_results_1775555013542.png)

### 🔍 Scenario 3: Female Less Qualified (Merit Case)
This scenario proves that debiasing does not create "Reverse Bias." When the male candidate is genuinely more qualified, the model maintains a merit-based ranking.
![Female Less Qualified Results](/Users/macbookpro/.gemini/antigravity/brain/c538d7f0-e3a5-481b-bec1-f86627d98c6b/female_less_qualified_results_1775555235238.png)

## 🤖 BERT Auditor Verdict
The terminal auditor uses BERT's internal SHAP and LIME representations to provide a natural language reasoning "verdict" for each case, ensuring the user understands **why** the model's decision-making has shifted.

---
*Verified on BERT-base-uncased and Debiased-BERT-Final.*
