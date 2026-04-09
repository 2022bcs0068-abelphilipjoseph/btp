# Decentralized Fairness Validation in Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ethereum](https://img.shields.io/badge/Ethereum-Sepolia-blue)](https://sepolia.etherscan.io/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

A unified framework for bias mitigation, topological explainability, and cryptographic verification across LLM architectures.

## ✨ Abstract
Transformer-based language models propagate structural biases, limiting their trustworthy deployment in high-stakes domains. This work presents a unified framework for bias mitigation, explainability, and cryptographic verification across diverse architectures, including bidirectional encoders (BERT) and generative decoders (Phi-3, Qwen). We advance bias evaluation from superficial probability metrics to intrinsic geometric analysis (Cosine Distance, WEAT) and utilize the Logit Lens for layer-wise bias analysis, tracking exactly where stereotypes form computationally. 

To resolve cross-architecture explainability challenges, we introduce tokenizer-agnostic "Linguistic Mega-Tokens" and Target-Self-Attention Masking, enabling mathematically consistent "Bias Gaze Snapshots" using SHAP, LIME, and LIG. Finally, we secure auditability using a Zero-Trust Decoupled Blockchain Architecture. By enforcing Cryptographic Role-Based Access Control (RBAC) via MetaMask-signed Solidity smart contracts, the system guarantees immutable, decentralized verification of fairness metrics. Ultimately, this establishes a robust, end-to-end pipeline for auditing and enforcing fairness in Large Language Models.

---

## 🏛️ Ecosystem Architecture

![Architecture Diagram](./arch_diagram.png)

---

## 📁 Repository Structure

The project is decoupled into micro-repositories, maintaining clear boundaries between the deep-learning backend and the Web3 infrastructure.

| Directory | Purpose |
| --- | --- |
| [`Agentic_BERT_Dashboard/`](./Agentic_BERT_Dashboard) | Core PyTorch/HuggingFace backend and Streamlit dashboard for generating SHAP, LIME, and LRP fairness explanations. |
| [`blockchain_ui_trial/`](./blockchain_ui_trial) | Web3 frontend portal bridging AI agent outputs to Ethereum via `ethers.js` and MetaMask. |
| [`contracts/`](./contracts) | Solidity Smart Contracts enabling Role-Based Access Control (RBAC) and immutable audit storage. |
| [`scripts/`](./scripts) | Hardhat deployment scripts for the Sepolia Testnet. |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10** via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- **Node.js ≥ 18** (for Hardhat smart contract compilation)
- **MetaMask** browser extension with Sepolia Testnet ETH
- **GPU server** (recommended) — BERT + SHAP/LIME/LIG inference is compute-intensive

### 1. Environment Setup
We use Conda instead of `venv` to ensure reliable CUDA/GPU driver compatibility on HPC servers.

```bash
conda create -n agentic_env python=3.10 -y
conda activate agentic_env
```

### 2. Install Python Dependencies
```bash
cd Agentic_BERT_Dashboard/explainability
pip install -r requirements.txt
pip install seaborn torchvision pandas    # additional deps not in requirements.txt
```

### 3. Place Model Weights
The dashboard requires pre-trained model weights that are **not** tracked by Git (blocked by `.gitignore` due to their size). Copy them into `Agentic_BERT_Dashboard/explainability/`:

```
explainability/
├── debiased_bert_final/       ← Fine-tuned BERT (~438 MB)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── Qwen_finetuned_merged/     ← Fine-tuned Qwen
    ├── config.json
    ├── model.safetensors
    └── ...
```

> See [`Agentic_BERT_Dashboard/README.md`](./Agentic_BERT_Dashboard) for the complete file listing and `scp` transfer instructions.

### 4. Launch the Agentic Dashboard
```bash
cd Agentic_BERT_Dashboard/explainability

# Secure binding — prevents PII exposure on shared networks
streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

### 5. Launch the Web3 Auditor Portal
In a separate terminal:
```bash
cd blockchain_ui_trial
python -m http.server 8000
```
> See [`blockchain_ui_trial/README.md`](./blockchain_ui_trial) for MetaMask connection instructions.

### 6. Deploy Smart Contracts (Optional)
Only required if deploying a fresh contract instance:
```bash
npm i
npx hardhat compile
npx hardhat run scripts/deploy.ts --network sepolia
```

---

## 🔗 Blockchain Integration Workflow

The dashboard and Web3 portal communicate via a **URL-Parameter Bridging** architecture, bypassing MetaMask's iframe security restrictions entirely.

```
 Streamlit Dashboard                    Auditor Portal                   Ethereum Sepolia
┌──────────────────┐              ┌──────────────────────┐           ┌──────────────────┐
│                  │              │                      │           │                  │
│ 1. Run bias      │              │ 3. URLSearchParams   │           │ 5. logAudit()    │
│    audit on      │  SHA-256     │    parses hash from  │  MetaMask │    commits hash  │
│    two resumes   ├──────────────▶    URL, pre-fills    ├───────────▶    permanently   │
│                  │  hash in URL │    the form          │  tx sign  │    on-chain      │
│ 2. hashlib       │              │                      │           │                  │
│    computes hash │              │ 4. Recruiter clicks  │           │ 6. Etherscan     │
│    natively      │              │    "Sign & Anchor"   │           │    receipt issued │
└──────────────────┘              └──────────────────────┘           └──────────────────┘
```

---

## 🔐 Security & Non-Repudiation Model

Our zero-trust architecture enforces multiple layers of security:

| Layer | Mechanism |
| --- | --- |
| **Data Privacy** | Raw resume text (PII) never touches the blockchain. Only the `SHA-256` hash of the AI's Markdown explanation is committed on-chain. |
| **Hash Integrity** | The hash is computed natively in Python (`hashlib`) on the server, not in the browser, preventing client-side tampering. |
| **Network Isolation** | Streamlit binds to `127.0.0.1` only; access is tunnelled via SSH port forwarding. |
| **XSS Prevention** | All user-supplied variables rendered via `innerHTML` in the Auditor Portal are sanitized through a custom `escapeHTML()` function. |
| **Access Control** | On-chain `onlyAuditor` modifier in Solidity ensures only wallet addresses authorized by the contract admin can write audit records. |
| **Non-Repudiation** | Every anchored audit generates a permanent Etherscan `tx` receipt binding the model version, hash, auditor wallet, and timestamp immutably together. |

---

## 🖥️ Remote / HPC Deployment

If running on a headless GPU server, use SSH port forwarding to access the dashboard from your local browser:

```bash
# On your local laptop
ssh -N -L 8501:127.0.0.1:8501 user@server_ip
```

Then open `http://127.0.0.1:8501` locally. Use `tmux` on the server to keep Streamlit alive across SSH disconnects:

```bash
# On the server
tmux new -s btp_dashboard
conda activate agentic_env
streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t btp_dashboard
```

> See [`Agentic_BERT_Dashboard/README.md`](./Agentic_BERT_Dashboard) for full deployment details.
