# Model Interpretability and Bias Analysis

This project provides tools for explaining and analyzing bias in large language models (LLMs) like BERT and Qwen. It uses various interpretability techniques, including Integrated Gradients (IG), Layer-wise Relevance Propagation (LRP) approximations, SHAP, LIME, and Counterfactual generation.

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you are working with Phi-3 specifically, you may also need to install dependencies from `requirements_phi3.txt`.*

## Model Setup

The model weights are not included in this repository due to their size. You need to place your fine-tuned model directories in the project root:

- BERT model: `./debiased_bert_final/`
- Qwen model: `./Qwen_finetuned_merged/`

Ensure these directories contain the `config.json`, `pytorch_model.bin` (or `model.safetensors`), and tokenizer files.

## Usage

### Generating the Combined Report

To generate a comprehensive interpretability report for a given sentence:

1. Open `main.py`.
2. Modify the `sentence` variable in the `main()` function if needed.
3. Run the script:
   ```bash
   python main.py
   ```
This will generate an `explanation_report.html` file containing the visualizations and explanations for both BERT and Qwen models.

### Debugging Individual Models

You can also run model-specific analysis using the debug scripts:
- Qwen: `python debug_lrp_qwen_v2.py`
- BERT: `python debug_lrp.py`
- Phi-3: `python debug_lrp_phi3.py`

## Project Structure

- `main.py`: The entry point for generating the combined HTML report.
- `debug_lrp_*.py`: Model-specific scripts for detailed Layer-wise Relevance analysis.
- `combine_reports.py`: Utility script for merging multiple HTML reports.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Configured to exclude models, cache, and temporary files.

## Acknowledgements

This work utilizes several interpretability libraries:
- [Captum](https://captum.ai/) (Integrated Gradients, LIG)
- [SHAP](https://github.com/slundberg/shap)
- [LIME](https://github.com/marcotcr/lime)
