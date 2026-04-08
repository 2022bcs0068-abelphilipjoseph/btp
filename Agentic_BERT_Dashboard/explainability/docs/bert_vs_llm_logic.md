# How BERT "Calculates" without "Prompting"

Your question is 100% correct: **You cannot "prompt" BERT like you prompt ChatGPT or Qwen.** BERT is an **Encoder**, not a **Decoder**. 

Here is how BERT "decides" who is better without a natural language prompt:

### 1. Fine-Tuning (The Training Step)
Unlike a general LLM that you just "talk to," BERT must be **Fine-tuned** on a specific task. 
- During development, your BERT model was shown thousands of examples like this:
    - *Example 1*: Resume with "Senior" + "10 yrs" + "AWS" -> **Label: 1 (Qualified)**
    - *Example 2*: Resume with "Junior" + "1 yr" + "HTML" -> **Label: 0 (Unqualified)**
- BERT adjusted its internal mathematical weights until it could accurately predict the Label from the text.

### 2. The Classification "Head"
BERT doesn't "write" an answer. It has a **Linear Layer** (a specialized math layer) at the very top of its architecture:
- This layer takes the final "Hidden State" of the `[CLS]` token (which represents the summary of the whole resume).
- It performs a **Matrix Multiplication** that results in two raw numbers (Logits).
- **The Decision**: Whichever number is higher (after Softmax) is the model's "choice."

### 3. Numerical Compatibility (Cross-Attention)
When you put the JD and Resume together: `JD [SEP] Resume`
- BERT's **Self-Attention** mechanism calculates a "Score" for how much each word in the JD relates to each word in the Resume.
- If the JD has "Security" and the Resume has "Firewall", BERT's internal weights (learned during pre-training) recognize they are semantically related.
- This creates a strong "Signal" through the layers that ultimately leads to a high "Qualified" score.

### 4. Why this matters for Bias:
Because BERT relies on **patterns it saw in the training data** rather than "reasoning":
- **Base BERT**: If the training data had many "Managers" named "John," it learned an accidental mathematical pattern: `Name: John -> +0.1 to score`.
- **Debiased BERT**: During your fine-tuning, you essentially "force" the model to unlearn that pattern by showing it thousands of diverse examples (e.g., both "John" and "Sarah" as Managers) until the weight for the names becomes nearly 0.

---
*Summary: BERT is a pattern-matching calculator, not a reasoning-based prompter.*
