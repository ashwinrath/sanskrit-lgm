"""
Sanskrit Inference Tokenizer - Minimal loader-only tokenizer.
"""

import re
from typing import List, Dict, Optional
import unicodedata


class SanskritTokenizer:
    DEVANAGARI_VOWELS = "अआइईउऊऋॠऌॡएऐओऔ"
    DEVANAGARI_CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"

    VOWEL_SIGNS = {
        "ा": "आ", "ि": "इ", "ी": "ई", "ु": "उ", "ू": "ऊ",
        "ृ": "ऋ", "ॄ": "ॠ", "ॢ": "ऌ", "ॣ": "ॡ",
        "े": "ए", "ै": "ऐ", "ो": "ओ", "ौ": "औ",
    }

    HALANT = "्"
    ANUSVARA = "ं"
    VISARGA = "ः"
    CHANDRABINDU = "ँ"

    PUNCTUATION = ["|", "||", "।", "॥", ".", "?", "!", ",", ";", "-",
                   "(", ")", '"', "'", ":", "=", "+", "/", "\\", "*",
                   "%", "&", "$", "#", "@", "[", "]", "{", "}", "<", ">",
                   "`", "~", "_"]

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPACE_TOKEN = "<SPACE>"
    NEWLINE_TOKEN = "<NL>"
    NEWLINE_MARKER = "<NEWLINE_MARKER>"

    def __init__(self, max_length=512, vocab_list=None, token_to_id=None):
        self.max_length = max_length

        if vocab_list is not None:
            self.vocab = vocab_list
            self.token_to_id = token_to_id if token_to_id else {t: i for i, t in enumerate(self.vocab)}
        elif token_to_id is not None:
            self.token_to_id = token_to_id
            self.vocab = [None] * len(token_to_id)
            for token, idx in token_to_id.items():
                if idx < len(self.vocab):
                    self.vocab[idx] = token
            self.vocab = [v for v in self.vocab if v is not None]
        else:
            raise ValueError("Need vocab_list or token_to_id")

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def normalize_text(self, text):
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^\S\n]+", " ", text)
        return text.strip()

    def decompose_word(self, word):
        phonemes = []
        if word == self.NEWLINE_MARKER:
            return [self.NEWLINE_TOKEN]

        i = 0
        while i < len(word):
            char = word[i]

            if i + 1 < len(word) and char == "|" and word[i + 1] == "|":
                phonemes.append("||")
                i += 2
                continue

            if char == "\n":
                phonemes.append(self.NEWLINE_TOKEN)
                i += 1
                continue

            if char in self.PUNCTUATION:
                phonemes.append(char)
                i += 1
                continue

            if i + 1 < len(word) and word[i + 1] == self.HALANT:
                phonemes.append(char + self.HALANT)
                i += 2
                continue

            if char in self.DEVANAGARI_CONSONANTS and i + 1 < len(word):
                next_char = word[i + 1]
                if next_char in self.VOWEL_SIGNS:
                    phonemes.append(char + next_char)
                    i += 2
                    continue

            if char in self.DEVANAGARI_VOWELS:
                phonemes.append(char)
                i += 1
                continue

            if char in self.DEVANAGARI_CONSONANTS:
                phonemes.append(char)
                i += 1
                continue

            if char in [self.ANUSVARA, self.VISARGA, self.CHANDRABINDU]:
                if phonemes:
                    phonemes[-1] += char
                else:
                    phonemes.append(char)
                i += 1
                continue

            i += 1

        return phonemes

    def encode(self, text, add_special_tokens=True, max_length=None, padding=True, truncation=True):
        if max_length is None:
            max_length = self.max_length

        text = self.normalize_text(text)
        text = text.replace("\n", f" {self.NEWLINE_MARKER} ")

        words = text.split()
        phonemes = []

        for i, word in enumerate(words):
            word_phonemes = self.decompose_word(word)
            phonemes.extend(word_phonemes)
            if i < len(words) - 1:
                phonemes.append(self.SPACE_TOKEN)

        if add_special_tokens:
            phonemes = [self.BOS_TOKEN] + phonemes + [self.EOS_TOKEN]

        if truncation and len(phonemes) > max_length:
            phonemes = phonemes[:max_length]
            if add_special_tokens:
                phonemes[-1] = self.EOS_TOKEN

        input_ids = [self.token_to_id.get(p, self.token_to_id.get(self.UNK_TOKEN, 0)) for p in phonemes]
        attention_mask = [1] * len(input_ids)

        if padding and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            pad_id = self.token_to_id.get(self.PAD_TOKEN, 0)
            input_ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {"input_ids": [int(x) for x in input_ids], "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
                continue
            tokens.append(token)

        text_parts = []
        current_word = []
        for token in tokens:
            if token == self.SPACE_TOKEN:
                if current_word:
                    text_parts.append("".join(current_word))
                    current_word = []
            elif token == self.NEWLINE_TOKEN:
                if current_word:
                    text_parts.append("".join(current_word))
                    current_word = []
                text_parts.append("\n")
            else:
                current_word.append(token)

        if current_word:
            text_parts.append("".join(current_word))

        text = " ".join(text_parts)
        text = text.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
        return text

    @property
    def vocab_size(self):
        return len(self.vocab)
