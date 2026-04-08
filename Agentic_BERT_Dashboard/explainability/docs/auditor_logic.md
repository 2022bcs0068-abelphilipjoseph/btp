# The BERT Auditor: What it is and How it works

The "**BERT Auditor**" is the brain of the dashboard that provides the final "Verdict." It isn't just a simple print statement; it's a decision-making layer built directly on top of BERT's internal data.

### 1. What "is" the Auditor?
Internally, it is a specialized logic function (`run_bert_audit`) in the code. It acts as an **Agentic Layer** because it doesn't just display data; it **interprets** it and provides a reasoning-based summary.

### 2. How does it work? (The 3-Step Process)

#### Step A: Difference Analysis (The "Gap")
First, it calculates the **Score Gap** for both the Base Model and the Debiased Model.
- **Formula**: `abs(Male Score - Female Score)`
- It looks for whether the gap has decreased. If the gap is smaller in the Debiased model, it triggers a **"✅ Bias Mitigation Detected"** signal.

#### Step B: Feature Attribution Audit (SHAP)
It looks "under the hood" of BERT using **SHAP (Shapley Additive Explanations)**.
- It extracts the **Top 10 words** that BERT focused on to make its decision.
- It scans these words for **Gender Keywords** (e.g., names like "Sarah", "John", or pronouns like "she", "he", "her").

#### Step C: Deterministic Reasoning
Based on the data from Steps A and B, it follows a set of rules to write the final summary:
- **If Bias is Gone**: "✅ BERT's internal reasoning shows it is now ignoring gender and focusing on skills like [Skill Tokens]."
- **If Bias Persists**: "⚠️ BERT's SHAP values show it is still being influenced by the name [Name] or pronoun [Pronoun]."
- **If Merit is High**: "BERT's decision-making is correctly driven by experience and skill-based tokens."

### 3. Summary of Inputs
| What it Sees | What it Does |
| :--- | :--- |
| **BERT Scores** | Measures fairness (Gap reduction). |
| **SHAP Tokens** | Detects if gender is still a "hidden" factor. |
| **LRP Maps** | Validates the "relevance flow" across layers. |

---
*Created by Agentic BERT Auditor.*
