import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, Lime
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from lime.lime_text import LimeTextExplainer
import shap
import seaborn as sns

def clean_token(token):
    """Removes tokenizer artifacts like 'Ġ' and '##'."""
    return token.replace('Ġ', '').replace('##', '').replace(' ', '')

def visualize_heatmap(data, x_labels, y_labels, title="Heatmap", xlabel="Tokens", ylabel="Layers"):
    """Generates a base64 encoded heatmap image."""
    plt.figure(figsize=(12, 8))
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

def visualize_attributions(attributions, tokens, title="LIG"):
    """Simple 1D heatmap for embeddings/single layer."""
    cleaned_tokens = [clean_token(t) for t in tokens]
    attributions = attributions / (np.max(np.abs(attributions)) + 1e-9)
    
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(attributions[np.newaxis, :], cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cleaned_tokens)))
    ax.set_xticklabels(cleaned_tokens, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title(title)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_str}"/>'

def debug_bert_gradients():
    print("--- Explaining BERT (Debiased) ---")
    model_path = "./debiased_bert_final"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    except Exception as e:
        print(f"Could not load local model: {e}. Using 'bert-base-uncased'.")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
    
    model.eval()
    
    sentence = "The manager interviewed the applicant because she was looking for a new role."
    print(f"Input: {sentence}")
    
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids']
    input_ids = inputs['input_ids']
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # helper to merge subwords
    def aggregate_subword_scores(tokens, scores):
        merged_tokens = []
        merged_scores = []
        
        current_token = ""
        current_score = 0.0
        
        for i, t in enumerate(tokens):
            # BERT uses '##' for subwords
            is_new_word = False
            
            if not t.startswith('##'):
                is_new_word = True
            
            # Punctuation/Special might be their own words, default logic handles it:
            # "##" is the only marker for continuation
            
            clean_t = t.replace('##', '')
            
            if is_new_word and current_token != "":
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
                current_token = clean_t
                current_score = scores[i]
            else:
                current_token += clean_t
                current_score += scores[i]
                
        if current_token != "":
            merged_tokens.append(current_token)
            merged_scores.append(current_score)
            
        return merged_tokens, np.array(merged_scores)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        target_idx = torch.argmax(logits).item()
        attentions = outputs.attentions # Tuple of (batch, heads, seq, seq)
    
    print(f"Predicted Class: {target_idx}")

    # --- 1. Layer-wise LIG (Approximating LRP) ---
    print("1. Computing Layer-wise LIG...")
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_embeds, attention_mask):
            return self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)[0]

    wrapper = ModelWrapper(model)
    # Get embeddings for LIG input
    inputs_embeds = model.bert.embeddings(input_ids)
    
    layer_attributions = []
    layer_names = []
    
    # helper for embeddings
    lig = LayerIntegratedGradients(wrapper, model.bert.embeddings)
    attr = lig.attribute(inputs=inputs_embeds, additional_forward_args=(inputs['attention_mask'],), target=target_idx, n_steps=20)
    score = attr.sum(dim=-1).squeeze(0).detach().numpy()
    
    agg_tokens, agg_score = aggregate_subword_scores(raw_tokens, score)
    
    layer_attributions.append(agg_score)
    layer_names.append("Embeddings")
    
    # Iterate through encoder layers
    for i, layer_module in enumerate(model.bert.encoder.layer):
        # Captum can attribute to any layer output
        # For BERT, layer_module.output is the output of the layer
        lig = LayerIntegratedGradients(wrapper, layer_module.output)
        try:
            attr = lig.attribute(inputs=inputs_embeds, additional_forward_args=(inputs['attention_mask'],), target=target_idx, n_steps=20)
            # Sum across hidden dimension
            score = attr.sum(dim=-1).squeeze(0).detach().numpy()
            
            _, agg_score = aggregate_subword_scores(raw_tokens, score)
            
            layer_attributions.append(agg_score)
            layer_names.append(f"Layer {i}")
        except Exception as e:
            print(f"Skipping Layer {i}: {e}")

    # Create Big Graph Heatmap (Layers x Tokens)
    # Stack attributions: shape (num_layers, seq_len)
    layer_attrs_np = np.array(layer_attributions)
    # Normalize globally or per layer? user wants "relevance". Global norm is better to compare layers.
    layer_attrs_np = layer_attrs_np / (np.max(np.abs(layer_attrs_np)) + 1e-9)
    
    lrp_heatmap = visualize_heatmap(layer_attrs_np, agg_tokens, layer_names, 
                                    title=f"Layer-wise Relevance (LIG) - Class {target_idx}",
                                    xlabel="Words", ylabel="Layers")

    # --- 2. Attention Heatmap ---
    print("2. Visualizing Attention...")
    # Average attention across heads for the LAST layer
    last_layer_attn = attentions[-1][0].mean(dim=0).detach().numpy() # (seq, seq)
    
    # Block aggregation function for 2D
    def aggregate_2d_matrix(matrix, tokens):
         boundaries = [0]
         for i, t in enumerate(tokens):
             if i == 0: continue
             if not t.startswith('##'):
                 boundaries.append(i)
         boundaries.append(len(tokens))
         
         new_dim = len(boundaries) - 1
         new_mat = np.zeros((new_dim, new_dim))
         new_labels = []
         
         for i in range(new_dim):
             start_i = boundaries[i]
             end_i = boundaries[i+1]
             word = "".join([t.replace('##', '') for t in tokens[start_i:end_i]])
             new_labels.append(word)
             for j in range(new_dim):
                 start_j = boundaries[j]
                 end_j = boundaries[j+1]
                 block = matrix[start_i:end_i, start_j:end_j]
                 new_mat[i, j] = block.sum() 
         return new_mat, new_labels

    agg_attn_mat, agg_labels = aggregate_2d_matrix(last_layer_attn, raw_tokens)

    attn_heatmap = visualize_heatmap(agg_attn_mat, agg_labels, agg_labels, 
                                     title="Average Self-Attention (Last Layer, Aggregated)",
                                     xlabel="Key (Words)", ylabel="Query (Words)")

    # --- 1.5. Attention Rollout ---
    print("1.5. Computing Attention Rollout...")
    def compute_attention_rollout(all_layer_matrices, start_layer=0):
        # all_layer_matrices: tuple of (batch, heads, seq, seq)
        # We average heads
        num_layers = len(all_layer_matrices)
        batch_size = all_layer_matrices[0].shape[0]
        seq_len = all_layer_matrices[0].shape[-1]
        
        # Initialize identity matrix
        joint_attention = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(all_layer_matrices[0].device)
        
        for i in range(start_layer, num_layers):
            # (batch, heads, seq, seq) -> (batch, seq, seq)
            att_mat = all_layer_matrices[i].mean(dim=1)
            
            # Add residual connection and normalize
            att_mat = 0.5 * att_mat + 0.5 * torch.eye(seq_len).unsqueeze(0).to(att_mat.device)
            att_mat = att_mat / att_mat.sum(dim=-1, keepdim=True)
            
            # Recursive multiplication: Rollout = A * Rollout_prev
            joint_attention = torch.bmm(att_mat, joint_attention)
            
        return joint_attention
    
    rollout_matrix = compute_attention_rollout(attentions)
    # For BERT, we want to see what embedding tokens flow to the [CLS] token (index 0)
    # The row at index 0 represents contributions to the [CLS] token representation at the final layer
    cls_rollout = rollout_matrix[0, 0, :].detach().numpy()
    
    # FIX: Zero out the [CLS] token's own contribution (index 0) to improve contrast for other tokens
    cls_rollout[0] = 0
    
    # AGGREGATE Rollout
    _, agg_rollout = aggregate_subword_scores(raw_tokens, cls_rollout)
    
    # Visualize Rollout as 1D heatmap (similar to attention but cumulative)
    rollout_heatmap = visualize_attributions(agg_rollout, agg_labels, title="Attention Rollout (Flow to [CLS])")

    # --- 3. LIME ---
    print("3. Running LIME...")
    def predict_proba(texts):
        toks = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
             l = model(**toks).logits
        return torch.softmax(l, dim=1).numpy()

    explainer = LimeTextExplainer(class_names=["Class 0", "Class 1"])
    exp = explainer.explain_instance(sentence, predict_proba, num_features=10)
    # Plot LIME
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

    # --- 4. SHAP ---
    print("4. Running SHAP...")
    # Wrap model for SHAP (takes text, outputs logits)
    def shap_predict(texts):
        # texts is a list/array of strings
        # list() conversion just in case inputs are numpy array
        toks = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            out = model(**toks)
        return out.logits.numpy()

    # Use a masker for text
    masker = shap.maskers.Text(tokenizer)
    # SHAP Explainer
    explainer_shap = shap.Explainer(shap_predict, masker)
    shap_values = explainer_shap([sentence])
    
    # Visualize SHAP
    # Since SHAP plots are javascript, we might default to a bar plot representation if static image needed.
    # We can extract values manually.
    shap_val_data = shap_values.values[0][:, target_idx] # (seq_len,)
    shap_tokens = shap_values.data[0] # list of tokens
    
    plt.figure(figsize=(12, 3))
    colors = ['red' if x > 0 else 'blue' for x in shap_val_data]
    plt.bar(range(len(shap_val_data)), shap_val_data, color=colors)
    plt.xticks(range(len(shap_val_data)), shap_tokens, rotation=45, ha='right')
    plt.title(f"SHAP Feature Importance (Class {target_idx})")
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str_shap = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    shap_html = f'<img src="data:image/png;base64,{img_str_shap}"/>'

    # --- 5. Counterfactual ---
    print("5. Running Counterfactual...")
    sentence_cf = sentence.replace("she", "he")
    inputs_cf = tokenizer(sentence_cf, return_tensors='pt')
    with torch.no_grad():
        logits_cf = model(**inputs_cf).logits
        probs_cf = torch.softmax(logits_cf, dim=1).numpy()[0]
        probs_orig = torch.softmax(logits, dim=1).numpy()[0]
    
    cf_diff = probs_cf[target_idx] - probs_orig[target_idx]
    cf_text = (f"Original Prob: {probs_orig[target_idx]:.4f}<br>"
               f"Counterfactual ('he'): {probs_cf[target_idx]:.4f}<br>"
               f"Difference: {cf_diff:.4f}")

    # --- Report ---
    report = f"""
    <div class='container' style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: #333;'>BERT (Debiased) Analysis</h2>
        <p><b>Input:</b> {sentence}</p>
        
        <h3>1. Layer-wise Relevance Propagation (Approximated via LIG)</h3>
        <p>This graph visualizes how information flows through the model layers. 
           The X-axis represents the input tokens, and the Y-axis represents the layers (0 is embeddings, 11 is top layer).
           <b>Interpretation:</b> Red hot-spots indicate tokens that strongly support the prediction at that specific layer.
           Stable red vertical lines suggest a token is consistently important throughout the network.</p>
        {lrp_heatmap}
        
        <h3>2. Attention Mechanism</h3>
        <p>This heatmap shows the self-attention weights averaged across all heads in the final layer.
           It represents "what the model is looking at" when processing each token.</p>
        {attn_heatmap}
        
        <h3>2.5 Attention Rollout</h3>
        <p>Attention Rollout (Abnar & Zuidema, 2020) recursively multiplies attention matrices to show how information flows from input tokens to the final <b>[CLS]</b> token embedding.</p>
        {rollout_heatmap}
        
        <h3>3. SHAP (SHapley Additive exPlanations)</h3>
        <p>SHAP values attribute the prediction to each feature (token) based on game theory.
           Red bars increase the probability of the predicted class, Blue bars decrease it.</p>
        {shap_html}
        
        <h3>4. LIME (Local Interpretable Model-agnostic Explanations)</h3>
        <p>LIME trains a local linear model to explain the prediction around this specific input.</p>
        {lime_html}
        
        <h3>5. Counterfactual Analysis</h3>
        <p>Impact of swapping pronouns on the prediction confidence.</p>
        <p><b>Sentence using Agregated Tokens:</b> {sentence_cf}</p>
        <p>{cf_text}</p>
    </div>
    """
    
    with open("explanation_report_bert.html", "w") as f:
        f.write(report)
    print("BERT report generated.")

if __name__ == "__main__":
    debug_bert_gradients()
