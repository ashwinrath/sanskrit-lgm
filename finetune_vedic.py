# save as: finetune_vedic.py
import torch
import json
import time
import math
from torch.utils.data import Dataset, DataLoader
from sanskrit_lgm import SanskritLGM, SanskritConfig
from train_lgm import SimpleTokenizer

class VedicDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=256):
        self.data = []
        for ex in examples:
            ids = tokenizer.encode(ex)
            if len(ids) > 3:
                self.data.append(ids[:max_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        max_len = 256
        pad_len = max_len - len(ids)
        input_ids = ids + [0] * pad_len
        labels = ids[1:] + [-100] * (pad_len + 1)
        labels = labels[:max_len]
        return torch.tensor(input_ids), torch.tensor(labels)

def train():
    device = torch.device("cuda")
    tokenizer = SimpleTokenizer("vocab.json")

    # Load Vedic examples
    with open("vedic_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split by ## headers into individual examples
    examples = [ex.strip() for ex in text.split("## ") if ex.strip()]
    # Add the header back and make each a complete example
    examples = ["## " + ex for ex in examples]

    # Also repeat examples to create more training data
    examples = examples * 50  # 40 examples × 50 = 2000 training samples
    print(f"Training examples: {len(examples)}")

    dataset = VedicDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    print(f"Batches: {len(loader)}")

    # Load pretrained model
    config = SanskritConfig.medium(vocab_size=tokenizer.vocab_size)
    model = SanskritLGM(config).to(device)
    ckpt = torch.load("lgm-medium-v2-best/model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded pretrained model")

    total, _ = model.count_params()
    print(f"Params: {total:,}")

    # Very low LR to not destroy Sanskrit knowledge
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # Train for 20 epochs (small dataset needs many passes)
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda"):
                _, loss = model(input_ids, labels=labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/20 | Loss: {avg:.4f}")

        # Sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            prompts = [
                "## सूत्र: योगसूत्रं रचयतु",
                "## यदि: संख्या",
                "## चक्रम्:",
                "## मान:",
            ]
            for p in prompts:
                ids = tokenizer.encode(p)[:-1]
                inp = torch.tensor([ids], device=device)
                gen = model.generate(inp, max_new_tokens=100, temperature=0.3, repetition_penalty=1.2)
                print(f"  [{p}]")
                print(f"  → {tokenizer.decode(gen[0].tolist())}")
            print()

    # Save
    model.save("./lgm-vedic")
    print("Saved to ./lgm-vedic")

    # Final test
    print("\n=== Final Vedic Code Generation ===\n")
    model.eval()
    prompts = [
        "## सूत्र: वर्गसूत्रं रचयतु",
        "## सूत्र: गुणनसूत्रं रचयतु",
        "## यदि: संख्या धनात्मक",
        "## चक्रम्: गुणन सारणी",
        "## मान: नाम",
    ]
    for p in prompts:
        ids = tokenizer.encode(p)[:-1]
        inp = torch.tensor([ids], device=device)
        gen = model.generate(inp, max_new_tokens=120, temperature=0.3, repetition_penalty=1.2)
        print(f"Prompt: {p}")
        print(f"Output: {tokenizer.decode(gen[0].tolist())}")
        print("-" * 50)

if __name__ == "__main__":
    train()