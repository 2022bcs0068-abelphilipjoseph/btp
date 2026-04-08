
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import LayerIntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import shap
import seaborn as sns
from lime.lime_text import LimeTextExplainer
import os

def clean_token(token):
    return token.replace('Ġ', '').replace(' ', '')

def visualize_heatmap(data, x_labels, y_labels, title="Heatmap", xlabel="Tokens", ylabel="Layers"):
    plt.figure(figsize=(14, 10))
    sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f'<img src="data:image/png;base64,{img_str}"/>'

def debug_llama3_explanations():
    print("--- Explaining Llama-3 ---")
    
    # Path to the full merged model
    model_path = "./wino_llama3_merged_v2/wino_llama3_FULL_MERGED"
    
    print("Loading Tokenizer...")
    try:
        # Try standard loading first
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e_auto:
        print(f"AutoTokenizer failed ({e_auto}). Trying PreTrainedTokenizerFast...")
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        except Exception as e_fast:
             print(f"Failed to load tokenizer: {e_fast}")
             return

    print("Loading Model...")
    try:
        # Load full model (no adapter needed as it is merged)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map="cpu", 
            output_attentions=True,
            attn_implementation="eager" # Important for Captum
        )
        model.config.use_cache = False # Disable cache for gradients
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    sentence = "The manager interviewed the applicant because she was looking for a new role."
    print(f"Input: {sentence}")
    
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]
    cleaned_tokens = [clean_token(t) for t in tokens]
    
    target_token_id = input_ids[0, -1].item()
    
    # --- 1. Layer-wise LIG ---
    print("1. Computing Layer-wise LIG...")
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_embeds):
            # Logits at -2 predict the token at -1
            return self.model(inputs_embeds=input_embeds).logits[:, -2, :]

    wrapper = ModelWrapper(model)
    
    embed_layer = None
    layers = []
    
    # Identify layers for Llama-3
    try:
        embed_layer = model.model.embed_tokens
        layers = model.model.layers
    except:
        print("Could not find embedding layer structure for Llama-3.")
    
    if embed_layer:
        inputs_embeds = embed_layer(input_ids)
        
        layer_attributions = []
        layer_names = []
        
        # Embeddings
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(wrapper)
        try:
            attr = ig.attribute(inputs=inputs_embeds, target=target_token_id, n_steps=20) 
            layer_attributions.append(attr.sum(dim=-1).squeeze(0).detach().float().numpy())
            layer_names.append("Embeddings")
        except Exception as e:
            print(f"Embeddings IG failed: {e}")
        
        # Layers (Stride to save time)
        stride = 2
        for i in range(0, len(layers), stride):
            layer = layers[i]
            print(f"Processing Layer {i}/{len(layers)}...")
            lig = LayerIntegratedGradients(wrapper, layer)
            try:
                 # attribute_to_layer_input=True is robust for skip connections
                attr = lig.attribute(inputs=inputs_embeds, target=target_token_id, n_steps=10, attribute_to_layer_input=True)
                layer_attributions.append(attr.sum(dim=-1).squeeze(0).detach().float().numpy())
                layer_names.append(f"Layer {i}")
            except Exception as e:
                print(f"Layer {i} LIG failed: {e}")
                
        # Heatmap
        if len(layer_attributions) > 0:
            layer_attrs_np = np.array(layer_attributions)
            # Normalize
            if layer_attrs_np.size > 0 and np.max(np.abs(layer_attrs_np)) > 0:
                layer_attrs_np = layer_attrs_np / np.max(np.abs(layer_attrs_np))
                
            lrp_heatmap = visualize_heatmap(layer_attrs_np, cleaned_tokens, layer_names, 
                                            title="Layer-wise Integrated Gradients (LIG) - Llama-3",
                                            xlabel="Tokens", ylabel="Layers")
        else:
            lrp_heatmap = "<p>LIG Failed.</p>"
    else:
        lrp_heatmap = "<p>LIG Failed (Structure not found).</p>"

    # --- 2. Attention ---
    print("2. Attention Heatmap...")
    with torch.no_grad():
        out = model(**inputs)
        attentions = out.attentions
        
    if attentions:
        last_layer_attn = attentions[-1][0].mean(dim=0).detach().float().numpy()
        attn_heatmap = visualize_heatmap(last_layer_attn, cleaned_tokens, cleaned_tokens,
                                         title="Average Self-Attention (Last Layer)",
                                         xlabel="Key", ylabel="Query")
    else:
        attn_heatmap = "<p>Attention not available.</p>"

    # --- 3. SHAP ---
    print("3. SHAP...")
    def shap_predict(texts):
        toks = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            out = model(**toks)
        try:
            return out.logits[:, -2, :].numpy()
        except IndexError:
            return out.logits[:, -1, :].numpy()

    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(shap_predict, masker)
        shap_values = explainer([sentence], max_evals=100)
        shap_val_data = shap_values.values[0][:, target_token_id]
        
        plt.figure(figsize=(12, 3))
        colors = ['red' if x > 0 else 'blue' for x in shap_val_data]
        plt.bar(range(len(shap_val_data)), shap_val_data, color=colors)
        plt.xticks(range(len(shap_val_data)), cleaned_tokens, rotation=45, ha='right')
        plt.title(f"SHAP Feature Importance (Target: {tokens[-1]})")
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str_shap = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        shap_html = f'<img src="data:image/png;base64,{img_str_shap}"/>'
    except Exception as e:
        print(f"SHAP Error: {e}")
        shap_html = f"<p>SHAP Error: {e}</p>"

    # --- 4. LIME ---
    print("4. LIME...")
    def lime_predict(texts):
        res = []
        for t in texts:
            toks = tokenizer(t, return_tensors='pt')
            with torch.no_grad():
                out = model(**toks)
            try:
                prob = torch.softmax(out.logits[0, -2], dim=0)[target_token_id].item()
            except IndexError:
                prob = torch.softmax(out.logits[0, -1], dim=0)[target_token_id].item()
            res.append([1-prob, prob])
        return np.array(res)
    
    try:
        explainer_lime = LimeTextExplainer(class_names=["Other", tokens[-1]])
        exp = explainer_lime.explain_instance(sentence, lime_predict, num_features=10, num_samples=20)
        lime_list = exp.as_list()
        lime_feats, lime_vals = zip(*lime_list)
        plt.figure(figsize=(10, 4))
        plt.barh(np.arange(len(lime_feats)), lime_vals, align='center')
        plt.yticks(np.arange(len(lime_feats)), lime_feats)
        plt.gca().invert_yaxis()
        plt.title("LIME Probability Contributions")
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str_lime = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        lime_html = f'<img src="data:image/png;base64,{img_str_lime}"/>'
    except Exception as e:
        lime_html = f"<p>LIME Error: {e}</p>"

    # --- 5. Counterfactual ---
    print("5. Counterfactual...")
    sentence_cf = sentence.replace("she", "he")
    inputs_cf = tokenizer(sentence_cf, return_tensors='pt')
    with torch.no_grad():
        out_orig = model(**inputs)
        prob_orig = torch.softmax(out_orig.logits[0, -2], dim=0)[target_token_id].item()
        out_cf = model(**inputs_cf)
        prob_cf = torch.softmax(out_cf.logits[0, -2], dim=0)[target_token_id].item()
    cf_diff = prob_cf - prob_orig
    cf_text = (f"Original Prob: {prob_orig:.4f}<br>Counterfactual: {prob_cf:.4f}<br>Diff: {cf_diff:.4f}")

    # --- Report ---
    report = f"""
    <div class='container' style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: #00008b;'>Llama-3 Analysis</h2>
        <p><b>Input:</b> {sentence}</p>
        
        <h3>1. Layer-wise Integrated Gradients (LIG)</h3>
        <p>This graph visualizes how information flows through the model layers.</p>
        {lrp_heatmap}
        
        <h3>2. Attention Heatmap</h3>
        {attn_heatmap}
        
        <h3>3. SHAP Feature Importance</h3>
        {shap_html}
        
        <h3>4. LIME Local Explanation</h3>
        {lime_html}
        
        <h3>5. Counterfactual Analysis</h3>
        <p>{cf_text}</p>
    </div>
    """
    
    with open("explanation_report_llama3.html", "w") as f:
        f.write(report)
    print("Llama-3 report generated.")

if __name__ == "__main__":
    debug_llama3_explanations()
