# Technical Explanation: Model Logic and Bias Detection

Based on your observation of the scores (Male 0.65 -> 0.25 vs. Female 0.45 -> 0.92), here is a detailed breakdown of how the model works.

### 1. How are we getting "who is the best"? (Scoring Logic)
The "Score" shown in the dashboard is the **Probability of Qualification**. 
- BERT outputs raw numbers called "logits" for two categories: **[0] Unqualified** and **[1] Qualified**.
- We use a mathematical function called **Softmax** to convert these into probabilities between 0 and 1.
- **Example**: If a candidate has a score of **0.92**, it means the model is **92% confident** that this candidate is a high match for the job description.
- The candidate with the highest probability is considered "the best" candidate.

### 2. How is BERT "doing it"? (Classification Mechanism)
BERT (Bidirectional Encoder Representations from Transformers) analyze the **context** of every word in relation to every other word:
- **Join Text Integration**: The model takes the Job Description (JD) and the Resume together, separated by a special token: `[JD] [SEP] [Resume]`.
- **Cross-Attention**: In the final layers, BERT's "Attention" mechanism compares JD requirements (e.g., "Python", "5 years") with Resume facts.
- **Semantic Matching**: It doesn't just look for exact word matches; it understands synonyms (e.g., "Distributed Systems" matching "K8s/AWS").

### 3. How is the model detecting Male vs. Female?
BERT identifies gender through **Tokenization**. Every word is broken into small pieces called "tokens":
- **Explicit Gender Markers**: Tokens like `[she]`, `[her]`, `[he]`, `[his]`.
- **Gendered Names**: Common names like `[sarah]`, `[jane]`, `[john]`, `[mark]` have learned associations in the base model.
- **Biased Associations**: In the **Base Model (Before)**, these gender tokens act as "shortcuts." For example, if the base model was trained on data where most successful candidates were male, it learns to give a "bonus" to the word `[mark]` and a "penalty" to `[sarah]`.

### 🔍 Proof in the Dashboard:
- **LRP & SHAP Tabs**: These reports literally *highlight* the words the model is looking at. 
- In the **Base Model**, you will see high importance (red/green) on names and pronouns.
- In the **Debiased Model**, you will see the importance shift away from names and towards skill-tokens like `[Kubernetes]` or `[Engineer]`.

---
*Created by Agentic BERT Auditor.*
