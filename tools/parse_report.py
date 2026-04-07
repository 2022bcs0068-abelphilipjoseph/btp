# tools/parse_report.py
import json
import re
import hashlib
from pathlib import Path

def parse_report_text(text):
    out = {
        "experiment_name": None,
        "integrated_gradients": [],
        "lime": [],
        "hap": [],
        "highlighted_text": None,
        "notes": ""
    }

    # Integrated Gradients block
    ig_match = re.search(r"Integrated Gradients Results[\\s\\S]*?Token Attributions.*?\\n(.*?)\\n\\n", text, re.M)
    if ig_match:
        ig_table = ig_match.group(1).strip()
        lines = [l.strip() for l in ig_table.splitlines() if l.strip()]
        for l in lines:
            if ',' in l:
                token, score = l.split(',',1)
                out["integrated_gradients"].append({"token": token.strip(), "attribution": float(score.strip())})

    # LIME block
    lime_match = re.search(r"LIME results\\n(.*?)\\n\\n", text, re.S)
    if lime_match:
        lines = lime_match.group(1).strip().splitlines()
        for l in lines:
            if ',' in l:
                cls, prob = l.split(',',1)
                out["lime"].append({"class": cls.strip(), "probability": float(prob.strip())})

    # Highlighted text
    hl_match = re.search(r"Text with highlighted words\\n\\n(.*?)\\n\\n", text, re.S)
    if hl_match:
        out["highlighted_text"] = hl_match.group(1).strip()

    # HAP block (look for contributions lines)
    hap_match = re.search(r"HAP Results Flow and Contributions[\\s\\S]*?\\n(.*)", text)
    if hap_match:
        hap_text = hap_match.group(1).strip()
        # find contribution lines like: Lower (Blue Arrows)\tw\tDecreases...
        hap_lines = [l for l in hap_text.splitlines() if l.strip()]
        for l in hap_lines:
            parts = [p.strip() for p in re.split(r'\t+', l)]
            if len(parts) >= 3:
                direction, word, contribution_type = parts[:3]
                out["hap"].append({"word": word, "direction": direction, "contribution_type": contribution_type})

    return out

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    src = Path("data/raw_reports/researcher_report.txt")
    dst = Path("data/artifacts/bias_summary.json")
    text = src.read_text(encoding='utf-8')
    parsed = parse_report_text(text)
    # Add experiment metadata
    parsed_meta = {
        "experiment_name": "bert_winobias_example",
        "dataset": {"name": "WinoBias_500", "sha256": ""},  # fill if available
        "model": {"name": "bert-base", "checkpoint_sha256": ""},
        "metrics": parsed,
        "notes": "Parsed from researcher_report.txt"
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(parsed_meta, indent=2), encoding='utf-8')
    print("Wrote:", dst)
    print("SHA256:", sha256_of_file(dst))

if __name__ == "__main__":
    main()
