# save as: train_lgm_pt.py
import torch
import torch.nn.functional as F
import json, time, math, sys
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/workspace/sanskrit")

from src.tokenizer import SanskritTokenizer
from sanskrit_lgm import SanskritLGM, SanskritConfig

class ShlokaDatasetPT(Dataset):
    def __init__(self, shlokas, tokenizer, max_len=256):
        self.data = []
        print("Tokenizing with ParamTatva tokenizer...")
        for i, shloka in enumerate(shlokas):
            enc = tokenizer.encode(shloka, padding=False, max_length=max_len, truncation=True)
            ids = enc['input_ids']
            if len(ids) > 3:
                self.data.append(ids)
            if (i + 1) % 20000 == 0:
                print(f"  {i+1:,} / {len(shlokas):,}")
        print(f"  Valid sequences: {len(self.data):,}")

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
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ParamTatva tokenizer
    with open("vocab.json", "r") as f:
        vocab_data = json.load(f)
    if isinstance(vocab_data, dict):
        tokenizer = SanskritTokenizer(token_to_id=vocab_data)
    else:
        tokenizer = SanskritTokenizer(vocab_list=vocab_data)
    print(f"Vocab: {tokenizer.vocab_size}")

    # Load corpus
    with open("sanskrit_corpus.txt", "r", encoding="utf-8") as f:
        shlokas = f.read().strip().split("\n")
    print(f"Shlokas: {len(shlokas):,}")

    dataset = ShlokaDatasetPT(shlokas, tokenizer, max_len=256)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_set):,}  Val: {len(val_set):,}")

    # Fresh model with medium config
    config = SanskritConfig.medium(vocab_size=tokenizer.vocab_size)
    model = SanskritLGM(config).to(device)
    total, _ = model.count_params()
    print(f"Model: {total:,} params ({config.num_layers}L, {config.num_heads}H, {config.hidden_dim}D)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(train_loader) * 10
    warmup = min(500, total_steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    eos_id = tokenizer.token_to_id.get("<EOS>", 3)
    bos_id = tokenizer.token_to_id.get("<BOS>", 2)

    prompts = [
        "धर्मो रक्षति",
        "सीता श्रीकृष्णस्य पत्नी",
        "अर्जुनः किम् अकरोत्",
        "कुरुक्षेत्रे समवेता युयुत्सवः",
    ]

    print(f"\nTraining for 10 epochs ({total_steps} steps)")
    print("=" * 60)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        start = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
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
            scheduler.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(f"  E{epoch+1} | {batch_idx+1}/{len(train_loader)} | Loss: {avg:.4f}")

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

        print(f"\n  Epoch {epoch+1}/10 | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            model.save("./lgm-paramtatva-best")
            print(f"  ✓ Saved (best)")

        # Generate samples
        for p in prompts:
            enc = tokenizer.encode(p, padding=False, add_special_tokens=True)
            ids = enc['input_ids'][:-1]  # remove EOS so model continues
            inp = torch.tensor([ids], device=device)
            gen = model.generate(inp, max_new_tokens=80, temperature=0.4, eos_token_id=eos_id)
            output = tokenizer.decode(gen[0].tolist())
            print(f"  [{p}] → {output}")
        print()

    model.save("./lgm-paramtatva-final")
    print(f"\nDone! Best val: {best_val:.4f}")

if __name__ == "__main__":
    train()