# Agentic Explainability Dashboard

This directory houses the core AI implementation utilizing Deep Learning Transformers (BERT, Qwen) to evaluate and explain predictive bias in text. It produces cryptographically verifiable Markdown audit reports that are anchored to the Ethereum Sepolia blockchain via the companion Web3 Auditor Portal.

## Features
- **Topological Bias Evaluation:** Computes intrinsic fairness metrics such as Cosine Distance embeddings and WEAT scores.
- **Agentic Explainability Integration:** Automatically aggregates LIME, LRP (Layer Integrated Gradients), and SHAP outputs to generate human-readable Markdown verdicts.
- **Cross-Architecture Scalability:** Supports bidirectional encoders (BERT) and generative causal decoders (Qwen) equivalently.
- **Blockchain-Ready Hash Export:** Natively computes SHA-256 hashes of audit reports using Python's `hashlib` and bridges them to the Web3 portal via URL parameters.

---

## Installation & Setup

### 1. Create the Conda Environment
We use Conda (via Miniconda or Anaconda) instead of `venv` to ensure reliable CUDA/GPU driver compatibility on HPC servers.

```bash
conda create -n agentic_env python=3.10 -y
conda activate agentic_env
```

### 2. Install Core Dependencies
```bash
cd Agentic_BERT_Dashboard/explainability
pip install -r requirements.txt
```

### 3. Install Additional Dependencies
The following packages are imported by `dashboard.py` but are **not** listed in `requirements.txt`. Install them manually:

```bash
pip install seaborn torchvision pandas
```

> **Note:** `torchvision` is a transitive dependency pulled in by some SHAP/Captum visualization utilities. `seaborn` is used for heatmap rendering. `pandas` is needed for data manipulation in the explainability pipeline.

---

## Model Weights Placement

The dashboard expects two sets of pre-trained model weights to be placed inside the `explainability/` directory. These files are **not** tracked by Git (blocked by `.gitignore`) and must be copied manually to any new deployment.

```
Agentic_BERT_Dashboard/
└── explainability/
    ├── dashboard.py
    ├── requirements.txt
    │
    ├── debiased_bert_final/          ← Fine-tuned BERT (Required)
    │   ├── config.json
    │   ├── model.safetensors         (~438 MB)
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── training_args.bin
    │   └── vocab.txt
    │
    └── Qwen_finetuned_merged/        ← Fine-tuned Qwen (Required)
        ├── config.json
        ├── model.safetensors         (varies)
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── ...
```

> **Tip:** If transferring weights to an HPC server, use `scp -r` or `rsync`:
> ```bash
> scp -r ./debiased_bert_final/ user@server:~/Desktop/BTP/Agentic_BERT_Dashboard/explainability/
> scp -r ./Qwen_finetuned_merged/ user@server:~/Desktop/BTP/Agentic_BERT_Dashboard/explainability/
> ```

---

## Secure Deployment (HPC / Remote Server)

### Running the Dashboard
Always bind Streamlit explicitly to `127.0.0.1` to prevent the dashboard from being exposed to the campus or public network:

```bash
conda activate agentic_env
cd Agentic_BERT_Dashboard/explainability

# Foreground (for debugging)
streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501

# Background (for persistent deployment)
nohup streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501 > streamlit.log 2>&1 &
```

> **Why `127.0.0.1`?** Without this flag, Streamlit defaults to `0.0.0.0`, which broadcasts the dashboard to every device on the network. Since the dashboard processes resume text, this would be a direct PII exposure vulnerability.

### SSH Port Forwarding (Access from Your Laptop)
On your **local laptop**, open a terminal and run:

```bash
ssh -N -L 8501:127.0.0.1:8501 user@server_ip
```

Then open `http://127.0.0.1:8501` in your browser. The dashboard renders locally, but all GPU inference executes on the remote server.

### Using tmux for Session Persistence
To prevent the dashboard from dying when your VPN or SSH connection drops:

```bash
# On the server
tmux new -s btp_dashboard

# Inside tmux, start the dashboard
conda activate agentic_env
streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501

# Detach: press Ctrl+B, then D
# Re-attach later:
tmux attach -t btp_dashboard
```

---

## Blockchain Integration Workflow

The dashboard connects to the Web3 Auditor Portal using a **URL-Parameter Bridging** architecture. This bypasses MetaMask's iframe security restrictions entirely.

```
┌─────────────────────────────┐
│  1. Recruiter runs audit    │
│     in Streamlit Dashboard  │
│                             │
│  dashboard.py generates     │
│  Markdown verdict           │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. Python hashlib computes │
│     SHA-256 of the report   │
│                             │
│  report_hash = "0x" +       │
│  hashlib.sha256(...).hex()  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. Streamlit renders a     │
│     redirect button with    │
│     hash embedded in URL:   │
│                             │
│  auditor_portal.html        │
│    ?hash=0xabc...           │
│    &modelVer=BERT-v2.1      │
└────────────┬────────────────┘
             │  (new browser tab)
             ▼
┌─────────────────────────────┐
│  4. Auditor Portal parses   │
│     URLSearchParams on load │
│     and pre-fills the hash  │
│                             │
│  Manual textarea is hidden  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5. MetaMask signs the      │
│     logAudit() transaction  │
│     on Sepolia via ethers.js│
│                             │
│  Hash is permanently        │
│  anchored on-chain          │
└─────────────────────────────┘
```

> **Security Guarantee:** Raw resume text (PII) never touches the blockchain. Only the cryptographic hash of the AI's explanation is committed, ensuring GDPR-compliant non-repudiation.
