"""
Step 6: Train a Sanskrit LGM on CPU
This is where the aha moment happens.

Expected time: ~15 minutes on modern CPU
Expected result: A model that generates Sanskrit-like text
"""
import torch
import json
import time
import math
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from sanskrit_lgm import SanskritLGM, SanskritConfig
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(ROOT, "data")


class SimpleTokenizer:
    """Simplified phoneme tokenizer for quick experiments."""
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
        self.CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
        self.HALANT = "्"
        self.VOWEL_SIGNS = {"ा":"आ","ि":"इ","ी":"ई","ु":"उ","ू":"ऊ",
                           "ृ":"ऋ","ॄ":"ॠ","े":"ए","ै":"ऐ","ो":"ओ","ौ":"औ"}
        self.VOWELS = "अआइईउऊऋॠऌॡएऐओऔ"
        self.SPECIALS = "ंःँ"

    def encode(self, text):
        tokens = [self.bos_id]
        words = text.strip().split()
        for wi, word in enumerate(words):
            i = 0
            while i < len(word):
                ch = word[i]
                if ch == "\n":
                    tokens.append(self.token_to_id.get("<NL>", self.unk_id)); i += 1; continue
                if ch in "।॥|.,;:?!-()\"'[]{}":
                    if i+1<len(word) and ch=="|" and word[i+1]=="|":
                        tokens.append(self.token_to_id.get("||", self.unk_id)); i += 2
                    else:
                        tokens.append(self.token_to_id.get(ch, self.unk_id)); i += 1
                    continue
                if ch in self.CONSONANTS and i+1<len(word) and word[i+1]==self.HALANT:
                    tokens.append(self.token_to_id.get(ch+self.HALANT, self.unk_id)); i += 2; continue
                if ch in self.CONSONANTS and i+1<len(word) and word[i+1] in self.VOWEL_SIGNS:
                    tokens.append(self.token_to_id.get(ch+word[i+1], self.unk_id)); i += 2; continue
                if ch in self.VOWELS:
                    tokens.append(self.token_to_id.get(ch, self.unk_id)); i += 1; continue
                if ch in self.CONSONANTS:
                    tokens.append(self.token_to_id.get(ch, self.unk_id)); i += 1; continue
                if ch in self.SPECIALS:
                    tokens.append(self.token_to_id.get(ch, self.unk_id)); i += 1; continue
                i += 1
            if wi < len(words) - 1:
                tokens.append(self.space_id)
        tokens.append(self.eos_id)
        return tokens

    def decode(self, ids):
        tokens = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if t in ("<PAD>","<BOS>","<EOS>","<UNK>"): continue
            if t == "<SPACE>": tokens.append(" ")
            elif t == "<NL>": tokens.append("\n")
            else: tokens.append(t)
        return "".join(tokens)


class ShlokaDataset(Dataset):
    def __init__(self, shlokas, tokenizer, max_len=128):
        self.data = []
        for s in shlokas:
            ids = tokenizer.encode(s)
            if len(ids) > 3:
                self.data.append(ids[:max_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        pad = 128 - len(ids)
        inp = ids + [0]*pad
        lab = ids[1:] + [-100]*(pad+1)
        return torch.tensor(inp[:128]), torch.tensor(lab[:128])


def train():
    device = torch.device("cpu")
    print("=" * 60)
    print("  Sanskrit LGM — CPU Training (Tiny Config)")
    print("  Expected time: ~15 minutes")
    print("=" * 60)

    tokenizer = SimpleTokenizer(os.path.join(DATA_DIR, "vocab.json"))
    print(f"\n  Vocab: {tokenizer.vocab_size} phonemes")

    with open(os.path.join(DATA_DIR, "sanskrit_corpus.txt"), "r", encoding="utf-8") as f:
        shlokas = f.read().strip().split("\n")

    # Use subset for CPU
    shlokas = shlokas[::3]  # every 3rd shloka
    print(f"  Shlokas: {len(shlokas):,} (subset for CPU)")

    dataset = ShlokaDataset(shlokas, tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"  Batches: {len(loader):,}")

    config = SanskritConfig.tiny(vocab_size=tokenizer.vocab_size)
    model = SanskritLGM(config)
    total, _ = model.count_params()
    print(f"  Model: {total:,} parameters")
    print(f"  Config: {config.num_layers}L, {config.num_heads}H, {config.hidden_dim}D")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"\n  Training for 3 epochs...\n")

    for epoch in range(3):
        model.train()
        epoch_loss = 0
        start = time.time()
        for batch_idx, (inp, lab) in enumerate(loader):
            _, loss = model(inp, labels=lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            if (batch_idx+1) % 50 == 0:
                print(f"    E{epoch+1} | {batch_idx+1}/{len(loader)} | Loss: {epoch_loss/(batch_idx+1):.4f}")

        avg = epoch_loss / len(loader)
        elapsed = time.time() - start
        print(f"\n  Epoch {epoch+1} | Loss: {avg:.4f} | Time: {elapsed:.0f}s")

        # Generate sample
        model.eval()
        prompt = "धर्मो रक्षति"
        ids = tokenizer.encode(prompt)[:-1]
        inp = torch.tensor([ids])
        gen = model.generate(inp, max_new_tokens=60, temperature=0.4, eos_token_id=tokenizer.eos_id)
        print(f"  Sample: {tokenizer.decode(gen[0].tolist())}\n")

    # Save
    save_dir = os.path.join(ROOT, "models", "lgm-tiny-cpu")
    model.save(save_dir)
    print(f"  ✓ Model saved to {save_dir}")

    # Final generation
    print(f"\n{'='*60}")
    print(f"  Generation Test")
    print(f"{'='*60}\n")

    model.eval()
    prompts = ["धर्मो रक्षति", "सीता", "अर्जुनः", "कुरुक्षेत्रे"]
    for p in prompts:
        ids = tokenizer.encode(p)[:-1]
        inp = torch.tensor([ids])
        gen = model.generate(inp, max_new_tokens=80, temperature=0.4)
        print(f"  {p} → {tokenizer.decode(gen[0].tolist())}")
        print()


if __name__ == "__main__":
    train()
