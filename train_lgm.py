# save as: train_lgm.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import time
import os
from sanskrit_lgm import SanskritLGM, SanskritConfig

# ============================================================
# 1. TOKENIZER (reuse ParamTatva's phoneme vocab)
# ============================================================

class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
        if isinstance(vocab_data, dict):
            self.token_to_id = vocab_data
        else:
            self.token_to_id = {t: i for i, t in enumerate(vocab_data)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.bos_id = self.token_to_id.get("<BOS>", 2)
        self.eos_id = self.token_to_id.get("<EOS>", 3)
        self.space_id = self.token_to_id.get("<SPACE>", 4)
        self.unk_id = self.token_to_id.get("<UNK>", 1)

        # Build char-to-phoneme mapping for fast encoding
        self.CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
        self.HALANT = "्"
        self.VOWEL_SIGNS = {
            "ा": "आ", "ि": "इ", "ी": "ई", "ु": "उ", "ू": "ऊ",
            "ृ": "ऋ", "ॄ": "ॠ", "े": "ए", "ै": "ऐ", "ो": "ओ", "ौ": "औ",
        }
        self.VOWELS = "अआइईउऊऋॠऌॡएऐओऔ"
        self.SPECIALS = "ंःँ"

    def encode(self, text):
        """Encode text to token IDs using phoneme decomposition."""
        tokens = [self.bos_id]
        words = text.strip().split()

        for wi, word in enumerate(words):
            i = 0
            while i < len(word):
                ch = word[i]

                # Newline
                if ch == "\n":
                    tokens.append(self.token_to_id.get("<NL>", self.unk_id))
                    i += 1
                    continue

                # Punctuation
                if ch in "।॥|.,;:?!-()\"'[]{}":
                    if i + 1 < len(word) and ch == "|" and word[i+1] == "|":
                        tokens.append(self.token_to_id.get("||", self.unk_id))
                        i += 2
                    else:
                        tokens.append(self.token_to_id.get(ch, self.unk_id))
                        i += 1
                    continue

                # Consonant + halant
                if ch in self.CONSONANTS and i + 1 < len(word) and word[i+1] == self.HALANT:
                    token = ch + self.HALANT
                    tokens.append(self.token_to_id.get(token, self.unk_id))
                    i += 2
                    continue

                # Consonant + vowel sign
                if ch in self.CONSONANTS and i + 1 < len(word) and word[i+1] in self.VOWEL_SIGNS:
                    token = ch + word[i+1]
                    tokens.append(self.token_to_id.get(token, self.unk_id))
                    i += 2
                    # Check for anusvara/visarga after
                    if i < len(word) and word[i] in self.SPECIALS:
                        prev = self.id_to_token.get(tokens[-1], "")
                        combined = prev + word[i]
                        if combined in self.token_to_id:
                            tokens[-1] = self.token_to_id[combined]
                        else:
                            tokens.append(self.token_to_id.get(word[i], self.unk_id))
                        i += 1
                    continue

                # Standalone consonant
                if ch in self.CONSONANTS:
                    tokens.append(self.token_to_id.get(ch, self.unk_id))
                    i += 1
                    if i < len(word) and word[i] in self.SPECIALS:
                        prev = self.id_to_token.get(tokens[-1], "")
                        combined = prev + word[i]
                        if combined in self.token_to_id:
                            tokens[-1] = self.token_to_id[combined]
                        else:
                            tokens.append(self.token_to_id.get(word[i], self.unk_id))
                        i += 1
                    continue

                # Vowel
                if ch in self.VOWELS:
                    tokens.append(self.token_to_id.get(ch, self.unk_id))
                    i += 1
                    continue

                # Special chars
                if ch in self.SPECIALS:
                    tokens.append(self.token_to_id.get(ch, self.unk_id))
                    i += 1
                    continue

                # Skip unknown
                i += 1

            # Add space between words
            if wi < len(words) - 1:
                tokens.append(self.space_id)

        tokens.append(self.eos_id)
        return tokens

    def decode(self, ids):
        tokens = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if t in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
                continue
            if t == "<SPACE>":
                tokens.append(" ")
            elif t == "<NL>":
                tokens.append("\n")
            else:
                tokens.append(t)
        return "".join(tokens)


# ============================================================
# 2. DATASET
# ============================================================

class ShlokaDataset(Dataset):
    def __init__(self, shlokas, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        print("Tokenizing shlokas...")
        for i, shloka in enumerate(shlokas):
            ids = tokenizer.encode(shloka)
            if len(ids) > 3:  # skip trivially short
                self.data.append(ids)
            if (i + 1) % 20000 == 0:
                print(f"  {i+1:,} / {len(shlokas):,}")

        print(f"  Total valid sequences: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx][:self.max_len]
        # Pad
        pad_len = self.max_len - len(ids)
        input_ids = ids + [0] * pad_len
        labels = ids[1:] + [-100] * (pad_len + 1)  # shift + ignore padding
        labels = labels[:self.max_len]
        return torch.tensor(input_ids), torch.tensor(labels)


# ============================================================
# 3. TRAINING LOOP
# ============================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    tokenizer = SimpleTokenizer("vocab.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Load corpus
    with open("sanskrit_corpus.txt", "r", encoding="utf-8") as f:
        shlokas = f.read().strip().split("\n")
    print(f"Total shlokas: {len(shlokas):,}")

    # Create dataset
    dataset = ShlokaDataset(shlokas, tokenizer, max_len=256)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_set):,}  Val: {len(val_set):,}")
    print(f"Train batches: {len(train_loader):,}")

    # Create model — MEDIUM config for A5000
    config = SanskritConfig.medium(vocab_size=tokenizer.vocab_size)
    model = SanskritLGM(config).to(device)
    total, trainable = model.count_params()
    print(f"\nModel: {total:,} parameters")
    print(f"Config: {config.num_layers} layers, {config.num_heads} heads, {config.hidden_dim} hidden\n")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

    # LR scheduler — cosine with warmup
    total_steps = len(train_loader) * 3  # 3 epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # Training
    print("=" * 60)
    print("  Training Started")
    print("=" * 60)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(3):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits, loss = model(input_ids, labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                avg = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                steps_per_sec = (batch_idx + 1) / elapsed
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1} | Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} | LR: {lr:.6f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda"):
                    _, loss = model(input_ids, labels=labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        epoch_time = time.time() - start_time
        train_loss = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch+1} complete | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {epoch_time:.0f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save("./lgm-medium-best")
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

        # Generate sample
        model.eval()
        prompt_text = "धर्मो रक्षति"
        prompt_ids = tokenizer.encode(prompt_text)
        prompt_ids = prompt_ids[:-1]  # remove EOS
        input_tensor = torch.tensor([prompt_ids], device=device)
        gen = model.generate(input_tensor, max_new_tokens=80, temperature=0.4, eos_token_id=tokenizer.eos_id)
        print(f"  Sample: {tokenizer.decode(gen[0].tolist())}")
        print()

    # Final save
    model.save("./lgm-medium-final")
    print("=" * 60)
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    # Final generation test
    print("\n=== Final Generation Test ===\n")
    model.eval()
    prompts = [
        "सीता श्रीकृष्णस्य पत्नी",
        "कुरुक्षेत्रे समवेता",
        "धर्मो रक्षति रक्षितः",
        "अर्जुनः किम् अकरोत्",
    ]
    for p in prompts:
        ids = tokenizer.encode(p)[:-1]
        inp = torch.tensor([ids], device=device)
        gen = model.generate(inp, max_new_tokens=100, temperature=0.4)
        print(f"Prompt: {p}")
        print(f"Output: {tokenizer.decode(gen[0].tolist())}")
        print("-" * 50)


if __name__ == "__main__":
    train()