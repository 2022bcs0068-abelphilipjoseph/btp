
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from captum.attr import LayerIntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import shap
import seaborn as sns
from lime.lime_text import LimeTextExplainer

def clean_token(token):
    # Phi-3 / Llama tokenizer usually uses ' ' (U+2581) for space
    return token.replace(' ', ' ').replace('Ġ', '').replace(' ', '')

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

def debug_phi3_explanations():
    print("--- Explaining Phi-3 (Debiased) ---")
    
    # Paths
    adapter_path = "./wino_phi3_debiased/wino_phi3_debiased"
    base_model_id = "microsoft/phi-3-mini-4k-instruct"
    
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    except:
        print("Fallback: Using locally cached tokenizer if available or fail.")
        return

    print("Loading Model...")
    try:
        # Use eager attention and disable cache to avoid complex graph issues
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            trust_remote_code=True, 
            device_map="cpu", 
            output_attentions=True,
            attn_implementation="eager" 
        )
        base_model.config.use_cache = False
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    sentence = "The manager interviewed the applicant because she was looking for a new role."
    print(f"Input: {sentence}")
    
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids']
    # Use convert_ids_to_tokens to see the subword markers (e.g. ' ')
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # helper to merge subwords
    def aggregate_subword_scores(tokens, scores):
        # tokens: list of strings (raw tokens with space markers)
        # scores: np.array of shape (len(tokens),)
        
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
                     try:
                        t_str = t.decode('latin-1') # fallback
                     except:
                        t_str = str(t)
            else:
                 t_str = str(t)
            
            # Llama/Phi-3 / SentencePiece style: ' ' (U+2581) starts a new word
            is_new_word = False
            
            # Check for SentencePiece underline or GPT-2 'Ġ'
            if t_str.startswith(' ') or t_str.startswith('Ġ') or t_str.startswith('\u2581') or t_str.startswith('\u0120'):
                is_new_word = True
            elif t_str.startswith('<') and t_str.endswith('>'): # Special tokens like <s>
                is_new_word = True
            
            # Heuristic: First token is new
            if i == 0:
                is_new_word = True
            
            clean_t = t_str.replace(' ', '').replace('Ġ', '').replace('\u2581', '').replace('\u0120', '')
            
            if is_new_word and current_token != "":
                # Push previous
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
                current_token = clean_t
                current_score = scores[i]
            else:
                # Merge
                current_token += clean_t
                current_score += scores[i]
                
        # Push last
        if current_token != "":
            merged_tokens.append(current_token)
            merged_scores.append(current_score)
            
        print(f"DEBUG: Merged into {len(merged_tokens)} words: {merged_tokens}")
        return merged_tokens, np.array(merged_scores)

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
    # Access embedding layer
    # Phi-3 structure usually base_model.model.model.embed_tokens
    try:
        embed_layer = model.base_model.model.model.embed_tokens
        layers = model.base_model.model.model.layers
    except:
        print("Could not find embedding layer structure.")
        embed_layer = None

    lrp_heatmap = "" 
    if embed_layer:
        inputs_embeds = embed_layer(input_ids)
        
        layer_attributions = []
        layer_names = []
        
        # Embeddings: Use IG instead of LIG to avoid bypass issue
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(wrapper)
        try:
            attr = ig.attribute(inputs=inputs_embeds, target=target_token_id, n_steps=20) 
            # Sum last dim to get scalar per token
            score = attr.sum(dim=-1).squeeze(0).detach().float().numpy()
            
            agg_tokens, agg_score = aggregate_subword_scores(raw_tokens, score)
            layer_attributions.append(agg_score)
            layer_names.append("Embeddings")
        except Exception as e:
            print(f"Embeddings IG failed: {e}")
        
        # Layers (sample every 3rd layer to speed up, Phi-3 has 32 layers)
        stride = 3 
        for i in range(0, len(layers), stride):
            layer = layers[i]
            print(f"Processing Layer {i}/{len(layers)}...")
            
            lig = LayerIntegratedGradients(wrapper, layer)
            try:
                # Reduced n_steps to 5 for speed on CPU
                attr = lig.attribute(inputs=inputs_embeds, target=target_token_id, n_steps=5, attribute_to_layer_input=True)
                score = attr.sum(dim=-1).squeeze(0).detach().float().numpy()
                
                # Aggregate
                _, agg_score = aggregate_subword_scores(raw_tokens, score)
                
                layer_attributions.append(agg_score)
                layer_names.append(f"Layer {i}")
            except Exception as e:
                print(f"Layer {i} LIG failed: {e}")

        # Heatmap
        if len(layer_attributions) > 0:
            layer_attrs_np = np.array(layer_attributions)
            # Normalize
            if layer_attrs_np.size > 0 and np.max(np.abs(layer_attrs_np)) > 0:
                layer_attrs_np = layer_attrs_np / np.max(np.abs(layer_attrs_np))
                
            lrp_heatmap = visualize_heatmap(layer_attrs_np, agg_tokens, layer_names, 
                                            title="Layer-wise Integrated Gradients (LIG) - Phi-3",
                                            xlabel="Words", ylabel="Layers")
        else:
            print("No layer attributions computed.")
            lrp_heatmap = "<p>LIG Calculation Failed (No layers succeeded).</p>"
    else:
        lrp_heatmap = "<p>LIG Failed (Embeddings not found).</p>"

    # --- 2. Attention ---
    print("2. Attention Heatmap...")
    with torch.no_grad():
        out = model(**inputs)
        attentions = out.attentions
        
    if attentions:
        last_layer_attn = attentions[-1][0].mean(dim=0).detach().float().numpy()
        # Aggregate Attention Matrix (2D)
        # We need to aggregate both rows and cols.
        # Simplification: Show Attention "Input Importance" (sum over rows for last token query?)
        # Or standard heatmap? If matrix 2D, we need to block-sum.
        
        # Let's do block aggregation for the matrix
        # 1. Get mapping from token_idx to word_idx
        word_mapping = [] # index -> word_index
        current_word_idx = -1
        # Re-run logic to get mapping
        
        # ... actually visualizing the full 2D aggregated matrix code is complex to squeeze here.
        # Let's visualize the "Attention to Last Token" which is a 1D vector (last row of matrix).
        # OR just aggregate the 1D vectors we used for others?
        # The user likely saw the 1D visualizations mostly.
        # BUT the Attention Heatmap is 2D.
        
        # Block aggregation function for 2D
        def aggregate_2d_matrix(matrix, tokens):
             boundaries = [0]
             for i, t in enumerate(tokens):
                 if i == 0: continue
                 
                 t_str = str(t)
                 if hasattr(t, 'decode'):
                     try: t_str = t.decode('utf-8')
                     except: t_str = str(t)
                 
                 is_new = False
                 if t_str.startswith(' ') or t_str.startswith('Ġ') or t_str.startswith('\u2581') or t_str.startswith('\u0120'):
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
                     if hasattr(t, 'decode'): 
                         try: s = t.decode('utf-8')
                         except: s = str(t)
                     s = s.replace(' ', '').replace('Ġ', '').replace('\u2581', '').replace('\u0120', '')
                     clean_segs.append(s)
                 word = "".join(clean_segs)
                 new_labels.append(word)
                 
                 for j in range(new_dim):
                     start_j = boundaries[j]
                     end_j = boundaries[j+1]
                     
                     # Sum the block
                     block = matrix[start_i:end_i, start_j:end_j]
                     new_mat[i, j] = block.sum() # Sum or mean? Sum of attention = total attention to the word
                     
             # Normalize rows? Attention usually sums to 1.
             # If we sum blocks, rows should roughly sum to 1 (if we aggregating queries too)
             return new_mat, new_labels
             
        agg_attn_mat, agg_labels = aggregate_2d_matrix(last_layer_attn, raw_tokens)
        
        attn_heatmap = visualize_heatmap(agg_attn_mat, agg_labels, agg_labels,
                                         title="Average Self-Attention (Last Layer, Aggregated)",
                                         xlabel="Key (Words)", ylabel="Query (Words)")
    else:
        attn_heatmap = "<p>Attention not available.</p>"

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
    
    # For GPT-style models (Phi-3), we want to see flow to the LAST token (target_token_id is predicted from last pos)
    # The last row represents contributions to the last token
    last_token_rollout = rollout_matrix[0, -1, :].detach().float().numpy()
    
    # FIX: Zero out the last token's own contribution to see what *past* tokens matter
    last_token_rollout[-1] = 0
    
    # AGGREGATE Rollout
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
        
        # Robust fallback using try-except
        try:
            return out.logits[:, -2, :].numpy()
        except IndexError:
            # Fallback for short sequences
            return out.logits[:, -1, :].numpy()

    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(shap_predict, masker)
        # Limit max evaluations
        shap_values = explainer([sentence], max_evals=100)
        
        shap_val_data = shap_values.values[0][:, target_token_id]
        
        plt.figure(figsize=(12, 3))
        colors = ['red' if x > 0 else 'blue' for x in shap_val_data]
        plt.bar(range(len(shap_val_data)), shap_val_data, color=colors)
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
    <div class='container' style='background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: #8b0000;'>Phi-3 (Debiased) Analysis</h2>
        <p><b>Input:</b> {sentence}</p>
        
        <h3>1. Layer-wise Integrated Gradients (LIG)</h3>
        <p>This graph visualizes how information flows through the model layers (LIG approximation).
           X-axis: Tokens, Y-axis: Layers.</p>
        {lrp_heatmap}
        
        <h3>2. Attention Heatmap</h3>
        <p>Self-attention weights averaged across heads in the last layer.</p>
        {attn_heatmap}
        
        <h3>2.5 Attention Rollout</h3>
        <p>Attention Rollout shows the recursive flow of information from input tokens to the final generated token.</p>
        {rollout_heatmap}
        
        <h3>3. SHAP Feature Importance</h3>
        {shap_html}
        
        <h3>4. LIME Local Explanation</h3>
        {lime_html}
        
        <h3>5. Counterfactual Analysis</h3>
        <p>Impact of swapping pronouns on the prediction confidence.</p>
        <p><b>Sentence using Agregated Tokens:</b> {sentence_cf}</p>
        <p>{cf_text}</p>
    </div>
    """
    
    with open("explanation_report_phi3.html", "w") as f:
        f.write(report)
    print("Phi-3 report generated.")

if __name__ == "__main__":
    debug_phi3_explanations()
