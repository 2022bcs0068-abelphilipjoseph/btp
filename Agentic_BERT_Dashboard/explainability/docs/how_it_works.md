# How this Dashboard Works (Workflow)

This dashboard is a specialized tool for auditing and correcting gender bias in AI-driven recruitment. Here is the step-by-step process of how it works:

### 1. User Input & Scenario Selection
You provide three pieces of text:
- **Job Description (JD)**: The requirements for the role.
- **Male Resume**: A candidate resume with male name/pronouns.
- **Female Resume**: A candidate resume with female name/pronouns.
- **Presets**: You can also use the **"Audit Scenarios"** in the sidebar to load pre-built cases (Equal Skills, Bias-Flip, etc.).

### 2. Dual-Model Inference (The "Comparison")
When you click **"Analyze Fairness"**, the dashboard sends your text to two different BERT models simultaneously:
- **Base Model (BERT-Base)**: The original, pre-trained model that likely contains historical gender biases.
- **Debiased Model**: The fine-tuned model that has been trained to ignore gendered tokens and focus strictly on professional merit.
- Both models calculate a **Qualification Score** (0.0 to 1.0) for both resumes.

### 3. Multi-Modal Explanation Generation
The dashboard doesn't just show scores; it generates four types of "XAI" (Explainable AI) reports to show **why** the models made those decisions:
- **LRP**: Traces the "flow of importance" through BERT's layers from the word tokens to the output.
- **SHAP**: Uses game theory to assign a numeric "contribution value" to each word.
- **LIME**: Builds a simple, local model to approximate BERT's behavior around your specific input.
- **Attention**: Extracts the actual "Focus Weights" from BERT's final self-attention head.

### 4. Agentic Auditor Reasoning
The final step is the **BERT Auditor Verdict**:
- A specialized internal agent analyzes the scores and the SHAP values.
- It detects if **Bias Mitigation** was successful (e.g., if a gap was closed).
- It generates a **Reasoning Verdict** in natural language, explaining if the model is still looking at gendered names/pronouns or if it has successfully shifted its focus to skills.

### 5. Final Output
You receive a side-by-side comparison table, a bar chart of the "Gap" reduction, and interactive tabs to explore the inner-workings of the debiased model.
