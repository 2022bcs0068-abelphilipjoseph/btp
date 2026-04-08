# Agentic Explainability Dashboard

This directory houses the core Artificial Intelligence implementation utilizing Deep Learning Transformers (e.g. BERT, Phi-3, Qwen) to evaluate and explain predictive bias in text.

## Features
- **Topological Bias Evaluation:** Computes intrinsic fairness metrics such as Cosine Distance embeddings and WEAT scores.
- **Agentic Explainability Integration:** Automatically aggregates LIME, LRP (Layer Integrated Gradients), and SHAP outputs to generate human-readable Markdown verdicts.
- **Cross-Architecture Scalability:** Supports bidirectional encoders (BERT) and generative causal decoders equivalently.

## Installation & Setup

1. Create and activate a Python virtual environment to avoid polluting global modules:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. Install the necessary dependencies:
    ```bash
    cd explainability
    pip install -r requirements.txt
    ```
    *Note: This utilizes `torch`, `transformers`, `captum`, `shap`, and `lime`.*

## Execution
Run the Streamlit application using the following command:
```bash
streamlit run dashboard.py
```

### GPU Support
If you have a CUDA-enabled GPU infrastructure available (such as a remote server), ensure your PyTorch build maps correctly to your hardware drivers. You can seamlessly port-forward the Streamlit dashboard (`localhost:8501`) via SSH to interact graphically on your local machine while executing on the remote GPU.
