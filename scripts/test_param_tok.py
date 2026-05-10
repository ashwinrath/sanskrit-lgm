# save as: test_paramtatva_tok.py
import torch
import json
import sys
sys.path.insert(0, "/workspace/sanskrit")

from src.tokenizer import SanskritTokenizer
from sanskrit_lgm import SanskritLGM, SanskritConfig

device = torch.device("cuda")

# Load tokenizer
with open("vocab.json", "r") as f:
    vocab_data = json.load(f)

if isinstance(vocab_data, dict):
    tokenizer = SanskritTokenizer(token_to_id=vocab_data)
else:
    tokenizer = SanskritTokenizer(vocab_list=vocab_data)

print(f"Vocab size: {tokenizer.vocab_size}")

# Test tokenization comparison
test = "धर्मो रक्षति रक्षितः"
encoded = tokenizer.encode(test, padding=False, add_special_tokens=False)
tokens = [tokenizer.id_to_token[i] for i in encoded['input_ids']]
print(f"\nText: {test}")
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)}")

# Test with a shloka
shloka = "कोन्वस्मिन् साम्प्रतं लोके गुणवान् कश्च वीर्यवान्"
enc = tokenizer.encode(shloka, padding=False, add_special_tokens=False)
tok = [tokenizer.id_to_token[i] for i in enc['input_ids']]
print(f"\nShloka: {shloka}")
print(f"Tokens ({len(tok)}): {tok}")
print(f"Decoded: {tokenizer.decode(enc['input_ids'], skip_special_tokens=False)}")