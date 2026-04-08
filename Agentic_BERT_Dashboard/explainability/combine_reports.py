
import os

def combine():
    bert_file = "explanation_report_bert.html"
    phi3_file = "explanation_report_phi3.html"
    
    content_bert = ""
    if os.path.exists(bert_file):
        with open(bert_file, "r") as f:
            content_bert = f.read()
            # Extract body content
            try:
                content_bert = content_bert.split("<body>")[1].split("</body>")[0]
            except:
                pass
    else:
        content_bert = "<p>BERT report missing.</p>"

    content_phi3 = ""
    phi3_file = "explanation_report_phi3.html"
    if os.path.exists(phi3_file):
        with open(phi3_file, "r") as f:
            content_phi3 = f.read()
            # If it has body tags, extract them, otherwise take full
            if "<body>" in content_phi3:
                 content_phi3 = content_phi3.split("<body>")[1].split("</body>")[0]
    else:
        content_phi3 = "<div style='padding: 20px; background: #ffe6e6; border: 1px solid #ffcccc; border-radius: 5px;'><h3>Phi-3 Analysis</h3><p><b>Status:</b> Report missing.</p></div>"

    content_qwen = ""
    qwen_file = "explanation_report_qwen.html"
    if os.path.exists(qwen_file):
        with open(qwen_file, "r") as f:
            content_qwen = f.read()
    else:
        content_qwen = "<p>Qwen report missing.</p>"

    content_llama3 = ""
    llama3_file = "explanation_report_llama3.html"
    if os.path.exists(llama3_file):
        with open(llama3_file, "r") as f:
            content_llama3 = f.read()
            if "<body>" in content_llama3:
                 content_llama3 = content_llama3.split("<body>")[1].split("</body>")[0]
    else:
        content_llama3 = "<div style='padding: 20px; background: #e6e6fa; border: 1px solid #ccccff; border-radius: 5px;'><h3>Llama-3 Analysis</h3><p><b>Status:</b> Failed. The provided model is 4-bit quantized and requires a GPU with `bitsandbytes` >= 0.43.1, which is not supported on this Mac CPU environment.</p></div>"
        
    final_html = f"""
    <html>
    <head><title>Combined Model Explanations</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }} 
        .container {{ border: 1px solid #ccc; padding: 1em; margin-bottom: 2em; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .model-section {{ margin-bottom: 4em; border-top: 2px solid #333; padding-top: 1em; }}
        h1 {{ color: #2c3e50; }}
        img {{ max-width: 100%; }}
    </style>
    </head>
    <body>
    <h1>Combined Model Explanation Report</h1>
    <p>This report compares the explanations for BERT (Debiased), Phi-3 (Debiased), Qwen (Finetuned), and Llama-3 (Finetuned) on the staffing domain example.</p>
    
    <div class='model-section'>
        {content_bert}
    </div>

    <div class='model-section'>
        {content_phi3}
    </div>

    <div class='model-section'>
        {content_qwen}
    </div>

    <div class='model-section'>
        {content_llama3}
    </div>
    
    </body>
    </html>
    """
    
    with open("explanation_report_final.html", "w") as f:
        f.write(final_html)
    print("Combined report saved to explanation_report_final.html")

if __name__ == "__main__":
    combine()
