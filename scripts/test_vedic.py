# save as: test_vedic.py
import torch
import json
from sanskrit_lgm import SanskritLGM, SanskritConfig
from train_lgm import SimpleTokenizer

device = torch.device("cuda")
tokenizer = SimpleTokenizer("vocab.json")

config = SanskritConfig.medium(vocab_size=tokenizer.vocab_size)
model = SanskritLGM(config).to(device)
ckpt = torch.load("lgm-medium-v2-best/model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompts = [
    "सूत्र योग",
    "मान फल",
    "यदि धर्मः अस्ति",
    "चक्रम् संख्या",
    "सूत्र क्रमगुणन",
]

for p in prompts:
    ids = tokenizer.encode(p)[:-1]
    inp = torch.tensor([ids], device=device)
    gen = model.generate(inp, max_new_tokens=80, temperature=0.3)
    print(f"Prompt: {p}")
    print(f"Output: {tokenizer.decode(gen[0].tolist())}")
    print("=" * 50)