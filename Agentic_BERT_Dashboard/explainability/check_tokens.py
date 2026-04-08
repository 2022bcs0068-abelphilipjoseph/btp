import torch
from transformers import AutoTokenizer

sentence = "The manager interviewed the applicant because she was looking for a new role."

def check_qwen():
    print("--- Qwen Tokens ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./Qwen_finetuned_merged", trust_remote_code=True)
        inputs = tokenizer(sentence, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(tokens)
        for t in tokens:
            print(f"'{t}'  Hex: {t.encode('utf-8').hex()}")
    except Exception as e:
        print(f"Qwen failed: {e}")

def check_phi3():
    print("\n--- Phi-3 Tokens ---")
    try:
        # debug_lrp_phi3.py uses "./wino_phi3_debiased" (deduced from file list)
        tokenizer = AutoTokenizer.from_pretrained("./wino_phi3_debiased", trust_remote_code=True)
        inputs = tokenizer(sentence, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"Type of first token: {type(tokens[0])}")
        print(tokens)
        for t in tokens:
             if isinstance(t, str):
                 print(f"'{t}' Hex: {t.encode('utf-8').hex()}")
             else:
                 print(f"{t} (Not a string)")
    except Exception as e:
        print(f"Phi-3 failed: {e}")

def check_bert():
    print("\n--- BERT Tokens ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./debiased_bert_final")
        inputs = tokenizer(sentence, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(tokens)
    except Exception as e:
        print(f"BERT failed: {e}")

if __name__ == "__main__":
    check_qwen()
    check_phi3()
    check_bert()
