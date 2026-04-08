
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import LayerIntegratedGradients, IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import shap
import seaborn as sns
from lime.lime_text import LimeTextExplainer

def clean_token(token):
    return token.replace('Ġ', '').replace('##', '').replace(' ', '')

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

def debug_qwen_explanations():
    print("--- Explaining Qwen (Finetuned) V2 ---")
    model_path = "./Qwen_finetuned_merged"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True, output_attentions=True)
    except Exception as e:
        print(f"Failed to load Qwen: {e}")
        return

    model.eval()
    
    sentence = "The manager interviewed the applicant because she was looking for a new role."
    print(f"Input: {sentence}")
    
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids']
    # Use convert_ids_to_tokens to see the subword markers
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # helper to merge subwords
    def aggregate_subword_scores(tokens, scores):
        merged_tokens = []
        merged_scores = []
        
        current_token = ""
        current_score = 0.0
        
        print(f"DEBUG: Aggregating {len(tokens)} tokens...")
        
        for i, t in enumerate(tokens):
            # Ensure t is string
            if hasattr(t, 'decode'): # is bytes
                 try:
                     t_str = t.decode('utf-8')
                 except:
                     t_str = str(t)
            else:
                 t_str = str(t)
                 
            # Qwen uses 'Ġ' (U+0120) or ' ' (U+2581) or similar.
            # We check both unicode codepoints directly to be safe.
            is_new_word = False
            
            # Debugging first few chars
            # print(f"Tok {i}: '{t_str}' starts with {ord(t_str[0]) if len(t_str)>0 else 'emp'}")

            if t_str.startswith('Ġ') or t_str.startswith(' ') or t_str.startswith('\u0120') or t_str.startswith('\u2581'):
                is_new_word = True
            elif t_str.startswith('<') and t_str.endswith('>'): # Special tokens
                is_new_word = True
            
            # Heuristic: If it looks like "The", it's a new word (no marker usually at start of seq)
            if i == 0:
                is_new_word = True # First token is always new
                
            # Clean
            clean_t = t_str.replace('Ġ', '').replace(' ', '').replace('\u0120', '').replace('\u2581', '')
            
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
            
        print(f"DEBUG: Merged into {len(merged_tokens)} words: {merged_tokens}")
        return merged_tokens, np.array(merged_scores)

    target_token_id = input_ids[0, -1].item()
    print(f"Target Token: {tokenizer.decode([target_token_id])}")

    # --- 1. Layer-wise LIG ---
    print("1. Computing Layer-wise LIG...")
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_embeds):
            # We want the prediction for the last token, which comes from the second-to-last position's logits
            return self.model(inputs_embeds=input_embeds).logits[:, -2, :]

    wrapper = ModelWrapper(model)
    
    try:
        embed_layer = model.model.embed_tokens
        layers = model.model.layers
    except:
        try:
            embed_layer = model.base_model.model.model.embed_tokens
            layers = model.base_model.model.model.layers
        except:
            embed_layer = model.transformer.wte
            layers = model.transformer.h

    inputs_embeds = embed_layer(input_ids)
    
    layer_attributions = []
    layer_names = []
    
    # Embeddings: Use IG
    ig = IntegratedGradients(wrapper)
    try:
        attr = ig.attribute(inputs=inputs_embeds, target=target_token_id, n_steps=20)
        score = attr.sum(dim=-1).squeeze(0).detach().float().numpy()
        
        agg_tokens, agg_score = aggregate_subword_scores(raw_tokens, score)
        
        layer_attributions.append(agg_score)
        layer_names.append("Embeddings")
    except Exception as e:
        print(f"IG Failed: {e}")
        # layer_names.append("Embeddings (Failed)")
        # Make sure shape matches agg_tokens length if we have it
        # Safest is to skip appending if failed
    
    # Layers: Use LIG (Sample layers)
    stride = 1
    if len(layers) > 12:
        stride = 2 
        
    for i in range(0, len(layers), stride):
        layer = layers[i]
        lig = LayerIntegratedGradients(wrapper, layer)
        try:
            attr = lig.attribute(inputs=inputs_embeds, target=target_token_id, n_steps=10)
            score = attr.sum(dim=-1).squeeze(0).detach().float().numpy()
            
            _, agg_score = aggregate_subword_scores(raw_tokens, score)
            
            layer_attributions.append(agg_score)
            layer_names.append(f"Layer {i}")
        except Exception as e:
            print(f"Skipping Layer {i}: {e}")
            
    layer_attrs_np = np.array(layer_attributions)
    if np.max(np.abs(layer_attrs_np)) > 0:
        layer_attrs_np = layer_attrs_np / np.max(np.abs(layer_attrs_np))
        
    lrp_heatmap = visualize_heatmap(layer_attrs_np, agg_tokens, layer_names, 
                                    title="Layer-wise Integrated Gradients (LIG) - Qwen",
                                    xlabel="Words", ylabel="Layers")

    # --- 2. Attention ---
    print("2. Attention Heatmap...")
    with torch.no_grad():
        out = model(**inputs)
        attentions = out.attentions
    
    if attentions:
        last_layer_attn = attentions[-1][0].mean(dim=0).detach().float().numpy()
        
        # Block aggregation function for 2D
        def aggregate_2d_matrix(matrix, tokens):
             boundaries = [0]
             for i, t in enumerate(tokens):
                 if i == 0: continue
                 # Replicate robust check
                 t_str = str(t)
                 if hasattr(t, 'decode'): t_str = t.decode('utf-8', errors='ignore')
                 
                 is_new = False
                 if t_str.startswith('Ġ') or t_str.startswith(' ') or t_str.startswith('\u0120') or t_str.startswith('\u2581'):
                     is_new = True
                 elif t_str.startswith('<') and t_str.endswith('>'):
                     is_new = True
                 
                 if is_new:
                     boundaries.append(i)
             boundaries.append(len(tokens))
             
             new_dim = len(boundaries) - 1
             new_mat = np.zeros((new_dim, new_dim))
             new_labels = []
             
             for i in range(new_dim):
                 start_i = boundaries[i]
                 end_i = boundaries[i+1]
                 # Label
                 segment = tokens[start_i:end_i]
                 clean_segs = []
                 for t in segment:
                     s = str(t)
                     if hasattr(t, 'decode'): s = t.decode('utf-8', errors='ignore')
                     s = s.replace('Ġ', '').replace(' ', '').replace('\u0120', '').replace('\u2581', '')
                     clean_segs.append(s)
                 word = "".join(clean_segs)
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
    else:
        attn_heatmap = "<p>No Attention Weights</p>"

    # --- 2.5 Attention Rollout ---
    print("2.5. Computing Attention Rollout...")
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
    
    # Compute Rollout
    rollout_matrix = compute_attention_rollout(attentions)
    
    # For GPT-style models (Qwen), we want to see flow to the LAST token (target_token_id is predicted from last pos)
    # The last row represents contributions to the last token
    last_token_rollout = rollout_matrix[0, -1, :].detach().float().numpy()
    
    # FIX: Zero out the last token's own contribution to see what *past* tokens matter
    last_token_rollout[-1] = 0
    
    # Aggregate Rollout
    _, agg_rollout = aggregate_subword_scores(raw_tokens, last_token_rollout)
    
    # Simple 1D visualization
    def visualize_1d_heatmap(data, tokens, title="Rollout"):
         # Normalize
         data = data / (np.max(np.abs(data)) + 1e-9)
         plt.figure(figsize=(12, 2))
         plt.imshow(data[np.newaxis, :], cmap='coolwarm', aspect='auto', vmin=0, vmax=1) # Attention is positive
         plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
         plt.yticks([])
         plt.title(title)
         plt.tight_layout()
         
         buf = io.BytesIO()
         plt.savefig(buf, format='png')
         buf.seek(0)
         img_str = base64.b64encode(buf.read()).decode('utf-8')
         plt.close()
         return f'<img src="data:image/png;base64,{img_str}"/>'

    rollout_heatmap = visualize_1d_heatmap(agg_rollout, agg_labels, title="Attention Rollout (Flow to Last Token)")

    # --- 3. SHAP ---
    print("3. SHAP...")
    def shap_predict(texts):
        toks = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            out = model(**toks)
        # Logits at -2 predict the token at -1
        try:
            return out.logits[:, -2, :].numpy()
        except IndexError:
            return out.logits[:, -1, :].numpy()

    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(shap_predict, masker)
        shap_values = explainer([sentence], max_evals=50)
        shap_val_data = shap_values.values[0][:, target_token_id]
        
        plt.figure(figsize=(12, 3))
        colors = ['red' if x > 0 else 'blue' for x in shap_val_data]
        plt.bar(range(len(shap_val_data)), shap_val_data, color=colors)
        # Use tokens from SHAP explainer
        plt.xticks(range(len(shap_val_data)), shap_values.data[0], rotation=45, ha='right')
        plt.title(f"SHAP Feature Importance (Target: {raw_tokens[-1]})")
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str_shap = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        shap_html = f'<img src="data:image/png;base64,{img_str_shap}"/>'
    except Exception as e:
        print(f"SHAP failed: {e}")
        shap_html = f"<p>SHAP calculation failed: {e}</p>"

    # --- 4. LIME ---
    print("4. LIME...")
    def lime_predict(texts):
        res = []
        for t in texts:
            toks = tokenizer(t, return_tensors='pt')
            with torch.no_grad():
                out = model(**toks)
            # Logits at -2 predict the token at -1
            try:
                prob = torch.softmax(out.logits[0, -2], dim=0)[target_token_id].item()
            except IndexError:
                prob = torch.softmax(out.logits[0, -1], dim=0)[target_token_id].item()
            res.append([1-prob, prob])
        return np.array(res)
    
    try:
        explainer_lime = LimeTextExplainer(class_names=["Other", raw_tokens[-1]])
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
        lime_html = f"<p>LIME failed: {e}</p>"

    # --- 5. Counterfactual ---
    print("5. Counterfactual...")
    sentence_cf = sentence.replace("she", "he")
    inputs_cf = tokenizer(sentence_cf, return_tensors='pt')
    with torch.no_grad():
        out_orig = model(**inputs)
        # Logits at -2 predict the token at -1
        prob_orig = torch.softmax(out_orig.logits[0, -2], dim=0)[target_token_id].item()
        
        out_cf = model(**inputs_cf)
        prob_cf = torch.softmax(out_cf.logits[0, -2], dim=0)[target_token_id].item()
        
    cf_diff = prob_cf - prob_orig
    cf_text = (f"Original Prob: {prob_orig:.4f}<br>Counterfactual: {prob_cf:.4f}<br>Diff: {cf_diff:.4f}")

    # --- Report ---
    report = f"""
    <div class='container' style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: #2c3e50;'>Qwen (Finetuned) Analysis</h2>
        <p><b>Input:</b> {sentence}</p>
        
        <h3>1. Layer-wise Integrated Gradients (LIG)</h3>
        {lrp_heatmap}
        
        <h3>2. Attention Heatmap</h3>
        {attn_heatmap}
        
        <h3>2.5 Attention Rollout</h3>
        <p>Attention Rollout shows the recursive flow of information from input tokens to the final generated token.</p>
        {rollout_heatmap}
        
        <h3>3. SHAP</h3>
        {shap_html}
        
        <h3>4. LIME</h3>
        {lime_html}
        
        <h3>5. Counterfactual Analysis</h3>
        <p>Impact of swapping pronouns on the prediction confidence.</p>
        <p><b>Sentence using Agregated Tokens:</b> {sentence_cf}</p>
        <p>{cf_text}</p>
    </div>
    """
    
    with open("explanation_report_qwen.html", "w") as f:
        f.write(report)
    print("Qwen report generated.")

if __name__ == "__main__":
    debug_qwen_explanations()
