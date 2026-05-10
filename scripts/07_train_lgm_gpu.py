"""
Step 7: Train Sanskrit LGM on GPU (medium config, 35M params)
Requires: NVIDIA GPU with 16+ GB VRAM (A5000, 4090, L40S, etc.)
Expected time: ~25 minutes for 10 epochs
"""
import torch
import json
import time
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from sanskrit_lgm import SanskritLGM, SanskritConfig
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(ROOT, "data")

# Import tokenizer — use ParamTatva if available, else simple
try:
    sys.path.insert(0, ROOT)
    from src.tokenizer import SanskritTokenizer
    USE_PT_TOKENIZER = True
    print("Using ParamTatva phoneme tokenizer ✓")
except ImportError:
    USE_PT_TOKENIZER = False
    print("ParamTatva tokenizer not found, using simple tokenizer")


class ShlokaDataset(Dataset):
    def __init__(self, shlokas, tokenizer, max_len=256, use_pt=False):
        self.data = []
        print("Tokenizing shlokas...")
        for i, s in enumerate(shlokas):
            if use_pt:
                enc = tokenizer.encode(s, padding=False, max_length=max_len, truncation=True)
                ids = enc['input_ids']
            else:
                ids = tokenizer.encode(s)[:max_len]
            if len(ids) > 3:
                self.data.append(ids)
            if (i+1) % 20000 == 0:
                print(f"  {i+1:,} / {len(shlokas):,}")
        print(f"  Valid: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        max_len = 256
        pad = max_len - len(ids)
        inp = ids + [0]*pad
        lab = ids[1:] + [-100]*(pad+1)
        return torch.tensor(inp[:max_len]), torch.tensor(lab[:max_len])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No GPU detected. Use scripts/06_train_lgm_cpu.py instead.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    if USE_PT_TOKENIZER:
        if isinstance(vocab_data, dict):
            tokenizer = SanskritTokenizer(token_to_id=vocab_data)
        else:
            tokenizer = SanskritTokenizer(vocab_list=vocab_data)
        vocab_size = tokenizer.vocab_size
        eos_id = tokenizer.token_to_id.get("<EOS>", 3)
    else:
        from scripts.train_lgm_cpu import SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_path)
        vocab_size = tokenizer.vocab_size
        eos_id = tokenizer.eos_id

    print(f"Vocab: {vocab_size}")

    # Load corpus
    with open(os.path.join(DATA_DIR, "sanskrit_corpus.txt"), "r", encoding="utf-8") as f:
        shlokas = f.read().strip().split("\n")
    print(f"Shlokas: {len(shlokas):,}")

    dataset = ShlokaDataset(shlokas, tokenizer, max_len=256, use_pt=USE_PT_TOKENIZER)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2, pin_memory=True)

    # Model
    config = SanskritConfig.medium(vocab_size=vocab_size)
    model = SanskritLGM(config).to(device)
    total, _ = model.count_params()
    print(f"Model: {total:,} params ({config.num_layers}L, {config.num_heads}H, {config.hidden_dim}D)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    warmup = min(500, total_steps // 10)

    def lr_fn(step):
        if step < warmup: return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    save_dir = os.path.join(ROOT, "models", "lgm-medium-gpu")

    prompts = ["धर्मो रक्षति", "सीता श्रीकृष्णस्य पत्नी", "अर्जुनः किम् अकरोत्", "कुरुक्षेत्रे समवेता"]

    print(f"\nTraining for {num_epochs} epochs")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start = time.time()

        for batch_idx, (inp, lab) in enumerate(train_loader):
            inp, lab = inp.to(device), lab.to(device)
            with torch.amp.autocast("cuda"):
                _, loss = model(inp, labels=lab)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()

            if (batch_idx+1) % 100 == 0:
                print(f"  E{epoch+1} | {batch_idx+1}/{len(train_loader)} | Loss: {epoch_loss/(batch_idx+1):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inp, lab in val_loader:
                inp, lab = inp.to(device), lab.to(device)
                with torch.amp.autocast("cuda"):
                    _, loss = model(inp, labels=lab)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        elapsed = time.time() - start
        train_loss = epoch_loss / len(train_loader)

        print(f"\n  Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            model.save(save_dir)
            print(f"  ✓ Best model saved")

        for p in prompts:
            if USE_PT_TOKENIZER:
                enc = tokenizer.encode(p, padding=False, add_special_tokens=True)
                ids = enc['input_ids'][:-1]
            else:
                ids = tokenizer.encode(p)[:-1]
            inp = torch.tensor([ids], device=device)
            gen = model.generate(inp, max_new_tokens=80, temperature=0.4, eos_token_id=eos_id)
            if USE_PT_TOKENIZER:
                output = tokenizer.decode(gen[0].tolist())
            else:
                output = tokenizer.decode(gen[0].tolist())
            print(f"  [{p}] → {output}")
        print()

    print(f"\nDone! Best val loss: {best_val:.4f}")
    print(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    train()
