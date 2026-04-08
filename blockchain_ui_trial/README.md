# Zero-Trust Web3 Auditor Portal

This directory contains the decentralized auditing interface bridging Web2 Machine Learning outputs to rigorous Web3 Non-Repudiation tracking systems.

## Purpose
The portal guarantees that outputs given by the AI (such as explaining why a certain decision was made) are not secretly rewritten by an enterprise after the fact. It achieves this by taking the cryptographically secure `SHA-256` hash of the AI's explanation and committing solely that hash to an immutable smart contract on the Ethereum Sepolia network. 

## Workflow
1. **Model Generates Report:** The Streamlit dashboard automatically computes the `SHA-256` payload locally inside its Python runtime to avoid PII (Personally Identifiable Information) exposure.
2. **Redirection Bridging:** Clicking "Sign & Anchor" inside the AI Dashboard injects the pre-computed hash securely via URL Parameters (`?hash=0x123...`) into the Auditor Portal.
3. **Decentralized Authorization:** `ethers.js` communicates with the active browser wallet (MetaMask) allowing the human operator to Cryptographically Sign the `logAudit()` smart contract function directly.

## Execution
Since this is a lightweight static Javascript architecture without a massive backend framework (to maximize transparency and fast auditing), you only need a generic HTTP server.

```bash
# Within the blockchain_ui_trial directory
python -m http.server 8000
```
*(Navigation to the portal happens automatically when redirected from the Streamlit UI, e.g. `http://127.0.0.1:8000/ai-audit-blockchain/auditor_portal.html`)*

### Prerequisites
- You must have the **MetaMask** Browser Extension installed.
- You must have an active wallet address supplied with Sepolia `ETH` (testnet funds) to cover the transaction gas fee for anchoring the report hash.
