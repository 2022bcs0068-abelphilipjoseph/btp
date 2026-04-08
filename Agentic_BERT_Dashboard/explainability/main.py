import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from captum.attr import IntegratedGradients, LayerIntegratedGradients, visualization, TokenReferenceBase, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from lime.lime_text import LimeTextExplainer
import shap
import re
import os
import base64
from io import BytesIO
import urllib.parse

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, attn_implementation="eager", output_attentions=True, trust_remote_code=True)
    return tokenizer, model

def load_bert_mlm(model_path):
    # Load the same BERT model but with a MaskedLM head
    # We assume the base weights are compatible or it's a standard BERT architecture
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    return model

def load_qwen_causal(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    return model

def visualize_attention(model, tokenizer, sentence, output_filename):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    attention = outputs.attentions[0].detach().numpy()  # Use the first layer's attention

    # Average attention heads
    attention = np.mean(attention[0], axis=0)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention, cmap='viridis')

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.colorbar(im)
    ax.set_title("Attention Heatmap")
    fig.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Attention heatmap saved to {output_filename}")

def explain_with_gradients(model, tokenizer, sentence):
    model.eval()

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs_embeds, attention_mask):
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]

    model_wrapper = ModelWrapper(model)
    ig = IntegratedGradients(model_wrapper)

    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    try:
        if hasattr(model, 'bert'):
            input_embeddings = model.bert.embeddings(input_ids)
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'): # For Qwen
            input_embeddings = model.model.embed_tokens(input_ids)
        else:
            raise AttributeError("Cannot find input embeddings layer")
    except AttributeError as e:
        warning_message = f"Warning: Could not find input embeddings. This part of gradient explanation is model-specific. Error: {e}"
        print(warning_message)
        return f"<p>{warning_message}</p>"


    attributions = ig.attribute(input_embeddings, additional_forward_args=(attention_mask,), target=0, internal_batch_size=1, n_steps=50) # n_steps can be adjusted

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    html = "<h4>Token Attributions (Integrated Gradients)</h4>"
    html += "<table><tr><th>Token</th><th>Attribution</th></tr>"
    for token, attribution in zip(tokens, attributions):
        html += f"<tr><td>{token}</td><td>{attribution.item():.4f}</td></tr>"
    html += "</table>"
    return html

def explain_with_lime(model, tokenizer, sentence, output_filename):
    explainer = LimeTextExplainer(class_names=['class_0', 'class_1'])

    def predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            return torch.softmax(logits, dim=1).numpy()

    explanation = explainer.explain_instance(sentence, predictor, num_features=10, num_samples=200)
    explanation.save_to_file(output_filename)
    print(f"LIME explanation saved to {output_filename}")

def explain_with_shap(model, tokenizer, sentence, output_filename):
    def predictor(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            return torch.softmax(logits, dim=1).numpy()

    explainer = shap.Explainer(predictor, tokenizer)
    shap_values = explainer([sentence])
    shap.save_html(output_filename, shap.force_plot(shap_values.base_values[0,0], shap_values.values[0,:,0], shap_values.data[0]))
    print(f"SHAP explanation saved to {output_filename}")

def generate_counterfactuals_bert(model, tokenizer, mlm_model, sentence, target_class=None):
    # 1. Get initial prediction and gradients to find important words
    model.eval()
    inputs = tokenizer(sentence, return_tensors='pt')
    probs = torch.softmax(model(**inputs).logits, dim=1)
    original_class = torch.argmax(probs).item()
    
    if target_class is None:
        target_class = 1 - original_class # Binary flip
        
    # Get importance using basic Integrated Gradients
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, inputs_embeds, attention_mask):
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]

    model_wrapper = ModelWrapper(model)
    ig = IntegratedGradients(model_wrapper)
    
    input_embeddings = model.bert.embeddings(inputs['input_ids'])
    attributions = ig.attribute(input_embeddings, additional_forward_args=(inputs['attention_mask'],), target=original_class, n_steps=20)
    attributions = attributions.sum(dim=-1).squeeze(0)
    
    # 2. Identify top K most important tokens (ignoring CLS/SEP)
    # We want tokens that contribute POSITIVELY to the original class (so removing them helps flip)
    # Or negatively to target class.
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_len = len(tokens)
    
    # Simple strategy: Mask the single most important token
    # Filter out special tokens
    special_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    valid_indices = [i for i in range(token_len) if inputs['input_ids'][0][i] not in special_ids]
    
    if not valid_indices:
        return "<p>No valid tokens to mask.</p>"

    # Find token with max attribution
    best_cf = None
    best_prob = 0.0
    
    html = "<h4>Counterfactuals (BERT Masking)</h4>"
    html += f"<p>Original: <b>{sentence}</b> (Class {original_class})</p>"
    html += "<ul>"
    
    # Sort indices by attribution magnitude
    sorted_indices = sorted(valid_indices, key=lambda i: attributions[i].item(), reverse=True)
    
    found = False
    for idx_to_mask in sorted_indices[:3]: # Try top 3 tokens
        masked_input_ids = inputs['input_ids'].clone()
        masked_input_ids[0][idx_to_mask] = tokenizer.mask_token_id
        
        # 3. Predict substitutes with MLM
        with torch.no_grad():
            outputs = mlm_model(masked_input_ids)
            predictions = outputs.logits[0, idx_to_mask].topk(5).indices
            
        for pred_token_id in predictions:
            if pred_token_id == inputs['input_ids'][0][idx_to_mask]:
                continue # Skip same word
                
            new_input_ids = inputs['input_ids'].clone()
            new_input_ids[0][idx_to_mask] = pred_token_id
            
            # Decode to text
            new_sentence = tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
            
            # 4. Verify with classifier
            new_inputs = tokenizer(new_sentence, return_tensors='pt')
            with torch.no_grad():
                new_logits = model(**new_inputs).logits
                new_probs = torch.softmax(new_logits, dim=1)
                new_class = torch.argmax(new_probs).item()
                
            if new_class == target_class:
                html += f"<li>Change <i>'{tokens[idx_to_mask]}'</i> to <b>'{tokenizer.decode([pred_token_id])}'</b> -> {new_sentence} (Class {new_class}, Prob: {new_probs[0][new_class]:.2f})</li>"
                found = True
    
    if not found:
        html += "<li>No single-token substitution flipped the prediction (Model classifier is likely uncertain/untrained).</li>"
        html += "<li><b>Top Candidates checked:</b></li><ul>"
        # Show what we tried
        for idx_to_mask in sorted_indices[:3]:
            masked_input_ids = inputs['input_ids'].clone()
            masked_input_ids[0][idx_to_mask] = tokenizer.mask_token_id
            with torch.no_grad():
                outputs = mlm_model(masked_input_ids)
                # Just take top 1 for display
                pred_token_id = outputs.logits[0, idx_to_mask].argmax().item()
            
            new_input_ids = inputs['input_ids'].clone()
            new_input_ids[0][idx_to_mask] = pred_token_id
            new_sentence = tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
            
            # Predict
            new_inputs = tokenizer(new_sentence, return_tensors='pt')
            with torch.no_grad():
                new_logits = model(**new_inputs).logits
                new_probs = torch.softmax(new_logits, dim=1)
                
            html += f"<li>Masked <i>'{tokens[idx_to_mask]}'</i> -> Filled <b>'{tokenizer.decode([pred_token_id])}'</b>: {new_sentence} (Class {torch.argmax(new_probs).item()}, Prob: {new_probs[0][torch.argmax(new_probs).item()]:.2f})</li>"
        html += "</ul>"
        
    html += "</ul>"
    return html

def generate_counterfactuals_qwen(model, tokenizer, sentence, classifier_model=None):
    # Prompt-based generation
    # Prompt-based generation with few-shot examples
    prompt = (
        f"Task: Rewrite the sentence to have the OPPOSITE sentiment (Positive <-> Negative). "
        f"Make minimal changes to the words."
        f"\n\nExample 1:"
        f"\nInput: The movie was fantastic."
        f"\nOutput: The movie was terrible."
        f"\n\nExample 2:"
        f"\nInput: I did not like the service."
        f"\nOutput: I loved the service."
        f"\n\nInput: {sentence}"
        f"\nOutput:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    try:
        outputs = model.generate(**inputs, max_new_tokens=30, num_return_sequences=1, do_sample=True, temperature=0.7)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the part after "Output:"
        if "Output:" in generated_text:
            cf_text = generated_text.split("Output:")[-1].strip().split("\n")[0]
        else:
            cf_text = generated_text.replace(prompt, "").strip()
            
        html = "<h4>Counterfactuals (Qwen Prompting)</h4>"
        html += f"<p>Original: {sentence}</p>"
        html += f"<p>Prompted Generation: <b>{cf_text}</b></p>"
        
        if classifier_model and cf_text.strip():
            try:
                # Predict on original
                inputs_orig = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs_orig = classifier_model(**inputs_orig)
                pred_orig_idx = torch.argmax(outputs_orig.logits, dim=1).item()
                
                # Predict on CF
                inputs_cf = tokenizer(cf_text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs_cf = classifier_model(**inputs_cf)
                pred_cf_idx = torch.argmax(outputs_cf.logits, dim=1).item()
                
                # Map labels if available
                id2label = getattr(classifier_model.config, 'id2label', {})
                label_orig = id2label.get(pred_orig_idx, str(pred_orig_idx))
                label_cf = id2label.get(pred_cf_idx, str(pred_cf_idx))
                
                html += f"<p>Original Prediction: {label_orig} &rarr; New Prediction: {label_cf}</p>"
                
                if pred_orig_idx != pred_cf_idx:
                    html += "<p style='color:green;'><b>SUCCESS: Prediction Flipped!</b></p>"
                else:
                    html += "<p style='color:orange;'><b>Note: Prediction did not flip.</b> Try adjusting the prompt or temperature.</p>"
            except Exception as e:
                html += f"<p>Error verifying flip: {e}</p>"
        
        html += "<p><i>Note: Validity depends on the generative model's capability.</i></p>"
        return html
    except Exception as e:
        return f"<p>Error generating Qwen counterfactual: {e}</p>"

def explain_with_lrp_layerwise(model, tokenizer, sentence, output_filename):
    # Using LayerIntegratedGradients to trace relevance layer-by-layer
    model.eval()
    inputs = tokenizer(sentence, return_tensors='pt')
    
    # Identify layers to trace
    layers = []
    layer_names = []
    
    if hasattr(model, 'bert'):
        # BERT layers
        embeddings = model.bert.embeddings
        encoder_layers = model.bert.encoder.layer
        layers = [encoder_layers[i] for i in range(len(encoder_layers))]
        layer_names = [f"Layer {i}" for i in range(len(layers))]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Qwen layers
        embeddings = model.model.embed_tokens
        layers = model.model.layers
        layer_names = [f"Layer {i}" for i in range(len(layers))]
    else:
        print("Model architecture not supported for custom Layer tracing.")
        return 
        
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, inputs_embeds, attention_mask):
             # We need to bypass embedding to use LIG
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]

    wrapper = ModelWrapper(model)
    
    # We will compute Attribution on each layer output
    # This acts as a proxy for "Relevance passing through this layer"
    # Note: Full LRP is complex; LayerIG shows "Contribution of this layer's output to final prediction"
    
    # We will compute Attribution on each layer output
    # This acts as a proxy for "Relevance passing through this layer"
    
    layer_relevance = [] # List of [seq_len] arrays
    
    # Get input embeddings for baseline
    try:
        if hasattr(model, 'bert'):
             input_embeds = model.bert.embeddings(inputs['input_ids'])
        else:
             input_embeds = model.model.embed_tokens(inputs['input_ids'])
        
        input_embeds.requires_grad_()
    except Exception as e:
        print(f"LRP Setup Error: {e}")
        return

    # Loop through layers individually to avoid graph issues with list
    for i, layer in enumerate(layers):
        try:
            # FIX: For BERT, target the .output submodule which returns a Tensor
            target_layer = layer
            if hasattr(model, 'bert'):
                target_layer = layer.output
                
            lig = LayerIntegratedGradients(wrapper, target_layer)
            
            # Use embeddings as input to allow differentiation
            attributions = lig.attribute(inputs=input_embeds, additional_forward_args=(inputs['attention_mask'],), target=0, n_steps=10)
            
            # Sum over hidden dim
            relevance = attributions.sum(dim=-1).squeeze(0).detach().numpy()
            layer_relevance.append(relevance)
        except Exception as e:
            print(f"Skipping Layer {i} due to LRP error: {e}")
            # Append zeros to maintain shape
            layer_relevance.append(np.zeros(inputs['input_ids'].shape[1]))

    if not layer_relevance:
        print("No layers attributed.")
        return
            
    layer_relevance = np.array(layer_relevance) # [num_layers, seq_len]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Visualization
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(layer_relevance, xticklabels=tokens, yticklabels=layer_names, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Layer-wise Relevance Propagation (Approximated via LIG)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"LRP heatmap saved to {output_filename}")
    except Exception as e:
        print(f"Error referencing LRP plot: {e}")
        # Create a dummy image to prevent report error
        plt.figure()
        plt.text(0.5, 0.5, f"LRP Error: {str(e)}", ha='center')
        plt.savefig(output_filename)
        plt.close()


def generate_combined_report(report_artifacts, output_filename):
    html = "<html><head><title>Model Explanation Report</title>"
    html += "<style> body { font-family: sans-serif; margin: 2em; } h1, h2, h3, h4 { color: #333; } table { border-collapse: collapse; margin-top: 1em; } th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; } tr:nth-child(even) { background-color: #f2f2f2; } img { max-width: 100%; height: auto; border: 1px solid #ddd; } iframe { width: 100%; height: 500px; border: 1px solid #ddd; } .container { border: 1px solid #ccc; padding: 1em; margin-bottom: 2em; border-radius: 5px; } </style>"
    html += "</head><body><h1>Model Explanation Report</h1>"

    for model_name, artifacts in report_artifacts.items():
        html += f"<div class='container'><h2>{model_name}</h2>"
        for explanation_name, artifact_path in artifacts.items():
            html += f"<h3>{explanation_name}</h3>"
            if artifact_path:
                if explanation_name == 'Attention':
                    with open(artifact_path, "rb") as f:
                        encoded_string = base64.b64encode(f.read()).decode()
                    html += f'<img src="data:image/png;base64,{encoded_string}">'
                elif explanation_name == 'Gradients':
                     html += artifact_path # It's already an HTML string
                elif explanation_name in ['LIME', 'SHAP', 'Counterfactuals']:
                    with open(artifact_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Use Base64 Data URI which is more robust than srcdoc or url-encoding for complex HTML/JS
                    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                    html += f'<iframe src="data:text/html;base64,{encoded_content}" style="width: 100%; height: 600px; border: 1px solid #ccc;"></iframe>'
                elif explanation_name == 'LRP':
                     with open(artifact_path, "rb") as f:
                        encoded_string = base64.b64encode(f.read()).decode()
                     html += f'<img src="data:image/png;base64,{encoded_string}">'
            else:
                html += "<p>Explanation not generated.</p>"
        html += "</div>"

    html += "</body></html>"

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Combined report saved to {output_filename}")


def main():
    bert_model_path = "./debiased_bert_final"
    qwen_model_path = "./Qwen_finetuned_merged/" 
    combined_report_file = "explanation_report.html"
    sentence = "this is a hellow world program."

    models = {"BERT": bert_model_path, "Qwen": qwen_model_path} #
    report_artifacts = {model_name: {} for model_name in models}
    temp_files = []
    
    # Load BERT MLM once
    print("\n--- Loading BERT MLM for Counterfactuals ---")
    try:
        bert_mlm = load_bert_mlm(bert_model_path)
    except Exception as e:
        print(f"Failed to load BERT MLM: {e}")
        bert_mlm = None
        
    # Load Qwen Causal once (if needed specifically for generation, but we can reuse the main one if we cast it? No, architecture differs)
    # Actually QwenForSequenceClassification might not support `generate` easily if head is different.
    # We'll rely on the Classifier model for Qwen for now, assuming it has a language model head underneath or use a hack.
    # QwenForSequenceClassification usually wraps QwenModel. 
    # Let's load Qwen Causal for generation specifically.
    print("\n--- Loading Qwen Causal for Generation ---")
    try:
        qwen_causal = load_qwen_causal(qwen_model_path)
    except Exception as e:
        print(f"Failed to load Qwen Causal: {e}")
        qwen_causal = None

    for model_name, model_path in models.items():
        print(f"\n--- Loading {model_name} model ---")
        try:
            tokenizer, model = load_model_and_tokenizer(model_path)
            if model_name == "Qwen":
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            print(f"{model_name} model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Could not load {model_name} model from '{model_path}'. Skipping. Error: {e}")
            if model_name in report_artifacts:
                del report_artifacts[model_name]
            continue

        print(f"\n--- Generating explanations for {model_name} model ---")
        
        # Attention
        attn_file = f"temp_attn_{model_name.lower()}.png"
        visualize_attention(model, tokenizer, sentence, attn_file)
        report_artifacts[model_name]['Attention'] = attn_file
        temp_files.append(attn_file)

        # Gradients
        print(f"Generating Gradients for {model_name}...")
        grad_html = explain_with_gradients(model, tokenizer, sentence)
        report_artifacts[model_name]['Gradients'] = grad_html

        # LIME
        lime_file = f"temp_lime_{model_name.lower()}.html"
        explain_with_lime(model, tokenizer, sentence, lime_file)
        report_artifacts[model_name]['LIME'] = lime_file
        temp_files.append(lime_file)

        # SHAP
        shap_file = f"temp_shap_{model_name.lower()}.html"
        explain_with_shap(model, tokenizer, sentence, shap_file)
        report_artifacts[model_name]['SHAP'] = shap_file
        temp_files.append(shap_file)
        
        # Counterfactuals
        print(f"Generating Counterfactuals for {model_name}...")
        cf_file = f"temp_cf_{model_name.lower()}.html"
        cf_html = ""
        if model_name == "BERT" and bert_mlm:
             cf_html = generate_counterfactuals_bert(model, tokenizer, bert_mlm, sentence)
        elif model_name == "Qwen" and qwen_causal:
             cf_html = generate_counterfactuals_qwen(qwen_causal, tokenizer, sentence, classifier_model=model)
        else:
             cf_html = "<p>Counterfactual generation pending model availability.</p>"
             
        with open(cf_file, 'w') as f:
            f.write(cf_html)
        report_artifacts[model_name]['Counterfactuals'] = cf_file
        temp_files.append(cf_file)
        
        # Layer-wise Relevance Propagation (LRP)
        print(f"Generating LRP for {model_name}...")
        lrp_file = f"temp_lrp_{model_name.lower()}.png"
        explain_with_lrp_layerwise(model, tokenizer, sentence, lrp_file)
        report_artifacts[model_name]['LRP'] = lrp_file
        temp_files.append(lrp_file)

    print("\n--- Generating combined report ---")
    generate_combined_report(report_artifacts, combined_report_file)

    # Clean up individual files
    print("\n--- Cleaning up temporary files ---")
    for f in temp_files:
        try:
            os.remove(f)
            print(f"Removed {f}")
        except OSError as e:
            print(f"Error removing file {f}: {e}")


if __name__ == "__main__":
    main()
