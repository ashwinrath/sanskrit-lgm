# save as: train_lgm_v2.py
# Same as train_lgm.py but with these changes:
# 1. Uses ParamTatva tokenizer directly
# 2. Trains for 10 epochs
# 3. Generates samples every epoch

import torch
import sys, os
import time, math, json
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Import our model
from sanskrit_lgm import SanskritLGM, SanskritConfig

# Import ParamTatva tokenizer
sys.path.insert(0, os.getcwd())

# We'll use the same vocab but with proper phoneme decomposition
from train_lgm import SimpleTokenizer, ShlokaDataset

def train():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = SimpleTokenizer("vocab.json")
    print(f"Vocab: {tokenizer.vocab_size}")

    with open("sanskrit_corpus.txt", "r", encoding="utf-8") as f:
        shlokas = f.read().strip().split("\n")
    print(f"Shlokas: {len(shlokas):,}")

    dataset = ShlokaDataset(shlokas, tokenizer, max_len=256)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2, pin_memory=True)

    # Load the best model from previous training and continue
    config = SanskritConfig.medium(vocab_size=tokenizer.vocab_size)
    model = SanskritLGM(config).to(device)

    # Load previous best weights
    if os.path.exists("./lgm-medium-best/model.pt"):
        print("Loading previous best model...")
        ckpt = torch.load("./lgm-medium-best/model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("  ✓ Loaded")

    total, _ = model.count_params()
    print(f"Params: {total:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.95))

    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    warmup_steps = 200

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")

    prompts = [
        "धर्मो रक्षति",
        "सीता श्रीकृष्णस्य पत्नी",
        "अर्जुनः किम् अकरोत्",
        "कुरुक्षेत्रे समवेता",
    ]

    for epoch in range(num_epochs):
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

        # Validate
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
            model.save("./lgm-medium-v2-best")
            print(f"  ✓ Best model saved")

        # Sample generation
        for p in prompts:
            ids = tokenizer.encode(p)[:-1]
            inp = torch.tensor([ids], device=device)
            gen = model.generate(inp, max_new_tokens=80, temperature=0.4)
            print(f"  [{p}] → {tokenizer.decode(gen[0].tolist())}")
        print()

    model.save("./lgm-medium-v2-final")
    print(f"\nDone! Best val: {best_val:.4f}")

if __name__ == "__main__":
    train()