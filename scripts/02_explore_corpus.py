"""
Step 2: Download and explore the Itihasa dataset
93,000 shlokas from the Ramayana and Mahabharata
"""
import json
import urllib.request
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

base = "https://raw.githubusercontent.com/rahular/itihasa/gh-pages/res"

print("=" * 60)
print("  Downloading Itihasa Dataset")
print("=" * 60)

for name in ["ramayana", "mahabharata"]:
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        print(f"  {name}.json already exists, skipping")
    else:
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(f"{base}/{name}.json", path)
        print(f"  ✓ {name}.json downloaded")

# Extract shlokas
def extract_shlokas(data):
    shlokas = []
    for vol_key, chapters in data.items():
        for chapter in chapters:
            for shloka in chapter['sn']:
                clean = shloka.strip()
                if clean:
                    shlokas.append(clean)
    return shlokas

with open(os.path.join(DATA_DIR, "ramayana.json"), "r", encoding="utf-8") as f:
    ram = extract_shlokas(json.load(f))
with open(os.path.join(DATA_DIR, "mahabharata.json"), "r", encoding="utf-8") as f:
    maha = extract_shlokas(json.load(f))

all_shlokas = ram + maha

print(f"\n  Ramayana shlokas:    {len(ram):,}")
print(f"  Mahabharata shlokas: {len(maha):,}")
print(f"  Total:               {len(all_shlokas):,}")

# Character analysis
corpus = "\n".join(all_shlokas)
chars = Counter(corpus)
devanagari = sum(c for ch, c in chars.items() if '\u0900' <= ch <= '\u097F' or ch in ' \n।॥')

print(f"\n  Corpus size: {len(corpus):,} characters")
print(f"  Unique chars: {len(chars)}")
print(f"  Devanagari purity: {100*devanagari/len(corpus):.1f}%")

# Save corpus
corpus_path = os.path.join(DATA_DIR, "sanskrit_corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_shlokas))

print(f"\n  ✓ Saved to {corpus_path}")

# Show samples
print(f"\n  === First 3 Shlokas ===\n")
for i, s in enumerate(all_shlokas[:3]):
    print(f"  {i+1}. {s}")
    print()

print("=" * 60)
print("  Next: python scripts/06_train_lgm_cpu.py")
print("=" * 60)
