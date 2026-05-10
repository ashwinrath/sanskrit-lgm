"""
Sanskrit LGM (Large Grammar Model) — Clean Text-Only Architecture
Based on ParamTatva's innovations, with improvements:

1. NO vision/video encoders (saves 55M params of overhead)
2. Repetition penalty built into generation
3. Rotary positional embeddings (RoPE) for better long-range context
4. Configurable model sizes from 1M to 50M
5. Clean training + inference in one file

Architecture:
  Input → PhonemeEmbed + SutraEmbed + PositionEmbed → Project(256)
    → N × [LayerNorm → MultiHeadAttn(+PratyaharaBias) → LayerNorm → FFN]
    → MaBridge Gate → FinalNorm → LMHead(vocab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class SanskritConfig:
    vocab_size: int = 1260
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: int = 1024
    max_seq_length: int = 512
    dropout: float = 0.1
    use_pratyahara_bias: bool = True
    use_sutra_embeddings: bool = True
    use_ma_bridge: bool = True
    use_rope: bool = True  # NEW: Rotary Position Embeddings

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0

    @staticmethod
    def tiny(vocab_size=1260):
        """~1M params — for quick experiments"""
        return SanskritConfig(
            vocab_size=vocab_size, hidden_dim=128,
            num_layers=4, num_heads=4, intermediate_dim=512,
            max_seq_length=256
        )

    @staticmethod
    def small(vocab_size=1260):
        """~7M params — matches ParamTatva decoder"""
        return SanskritConfig(
            vocab_size=vocab_size, hidden_dim=256,
            num_layers=6, num_heads=8, intermediate_dim=1024,
            max_seq_length=512
        )

    @staticmethod
    def medium(vocab_size=1260):
        """~25M params — bigger, better quality"""
        return SanskritConfig(
            vocab_size=vocab_size, hidden_dim=512,
            num_layers=8, num_heads=8, intermediate_dim=2048,
            max_seq_length=1024
        )

    @staticmethod
    def large(vocab_size=1260):
        """~50M params — max for 32GB CPU"""
        return SanskritConfig(
            vocab_size=vocab_size, hidden_dim=768,
            num_layers=12, num_heads=12, intermediate_dim=3072,
            max_seq_length=1024
        )


# ============================================================
# 2. ROTARY POSITION EMBEDDINGS (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):
    """RoPE — better than absolute positional embeddings for long sequences."""

    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================
# 3. SANSKRIT-SPECIFIC EMBEDDINGS
# ============================================================

class SanskritEmbedding(nn.Module):
    """
    Triple embedding: phoneme + sutra_index + position_in_sutra
    Encodes Panini's phonological classification directly.
    """

    def __init__(self, config: SanskritConfig):
        super().__init__()
        self.config = config
        self.phoneme_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        if config.use_sutra_embeddings:
            self.sutra_embeddings = nn.Embedding(15, config.hidden_dim)   # 14 sutras + null
            self.position_embeddings = nn.Embedding(11, config.hidden_dim) # max 10 positions + null
            self.projection = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
            self.register_buffer("_sutra_lookup", torch.zeros(config.vocab_size, dtype=torch.long))
            self.register_buffer("_position_lookup", torch.zeros(config.vocab_size, dtype=torch.long))
        else:
            self.projection = None

        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.phoneme_embeddings.weight, std=0.02)
        if self.config.use_sutra_embeddings:
            nn.init.normal_(self.sutra_embeddings.weight, std=0.02)
            nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        phoneme_emb = self.phoneme_embeddings(input_ids)

        if not self.config.use_sutra_embeddings:
            return self.dropout(phoneme_emb)

        sutra_idx = self._sutra_lookup[input_ids]
        position_idx = self._position_lookup[input_ids]
        sutra_emb = self.sutra_embeddings(sutra_idx)
        position_emb = self.position_embeddings(position_idx)

        combined = torch.cat([phoneme_emb, sutra_emb, position_emb], dim=-1)
        output = self.projection(combined)
        return self.dropout(output)


# ============================================================
# 4. PRATYAHARA ATTENTION BIAS
# ============================================================

class PratyaharaBias(nn.Module):
    """
    Phonological relationship bias for attention.
    Pre-wires which sounds attend to each other based on Panini's pratyahara groups.
    """

    def __init__(self, num_heads: int, vocab_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.register_buffer("pratyahara_matrix", torch.zeros(vocab_size, vocab_size))
        self.bias_scale = nn.Parameter(torch.ones(num_heads) * 0.1)

    def forward(self, phoneme_ids: torch.Tensor, attn_scores: torch.Tensor) -> torch.Tensor:
        batch, seq_len = phoneme_ids.shape
        vocab_size = self.pratyahara_matrix.size(0)
        ids_clamped = torch.clamp(phoneme_ids, 0, vocab_size - 1)

        idx_i = ids_clamped.unsqueeze(2).expand(batch, seq_len, seq_len)
        idx_j = ids_clamped.unsqueeze(1).expand(batch, seq_len, seq_len)
        relationships = self.pratyahara_matrix[idx_i.reshape(-1), idx_j.reshape(-1)]
        relationships = relationships.view(batch, seq_len, seq_len)

        bias = relationships.unsqueeze(1) * self.bias_scale.view(1, self.num_heads, 1, 1)
        return attn_scores + bias


# ============================================================
# 5. TRANSFORMER BLOCK
# ============================================================

class SanskritTransformerBlock(nn.Module):

    def __init__(self, config: SanskritConfig, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_dim // config.num_heads
        self.rope = rope

        # Self attention
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Pratyahara bias
        if config.use_pratyahara_bias:
            self.pratyahara = PratyaharaBias(config.num_heads, config.vocab_size)
        else:
            self.pratyahara = None

        # FFN with SwiGLU activation (better than GELU for small models)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn_gate = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.ffn_up = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.ffn_down = nn.Linear(config.intermediate_dim, config.hidden_dim)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, phoneme_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Self-attention with pre-norm
        residual = x
        x = self.attn_norm(x)

        q = self.q_proj(x).view(batch, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope is not None:
            cos, sin = self.rope(seq_len, x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Pratyahara bias
        if self.pratyahara is not None and phoneme_ids is not None:
            scores = self.pratyahara(phoneme_ids, scores)

        # Causal mask
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.hidden_dim)
        out = self.out_proj(out)
        x = residual + out

        # FFN with SwiGLU
        residual = x
        x = self.ffn_norm(x)
        gate = F.silu(self.ffn_gate(x))
        up = self.ffn_up(x)
        x = self.ffn_down(gate * up)
        x = self.ffn_dropout(x)
        x = residual + x

        return x


# ============================================================
# 6. MA-BRIDGE NORMALIZATION
# ============================================================

class MaBridge(nn.Module):
    """Gated normalization — grammatical filter on output."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * torch.sigmoid(self.gate(x))


# ============================================================
# 7. THE FULL MODEL
# ============================================================

class SanskritLGM(nn.Module):
    """
    Sanskrit Large Grammar Model — text-only, clean, efficient.
    """

    def __init__(self, config: SanskritConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = SanskritEmbedding(config)

        # RoPE
        self.rope = RotaryEmbedding(config.hidden_dim // config.num_heads, config.max_seq_length) if config.use_rope else None

        # Transformer blocks
        self.layers = nn.ModuleList([
            SanskritTransformerBlock(config, self.rope) for _ in range(config.num_layers)
        ])

        # Ma-Bridge
        self.ma_bridge = MaBridge(config.hidden_dim) if config.use_ma_bridge else None

        # Output
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embeddings.phoneme_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, phoneme_ids=input_ids)

        if self.ma_bridge is not None:
            x = self.ma_bridge(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.4,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        eos_token_id: Optional[int] = 3,
    ) -> torch.Tensor:
        """
        Improved generation with:
        - Repetition penalty (fixes the looping problem)
        - Top-p (nucleus) sampling
        - EOS detection
        - Context window management
        """
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate to max_seq_length if needed
                context = generated[:, -self.config.max_seq_length:]

                logits, _ = self.forward(context)
                next_logits = logits[:, -1, :] / temperature

                # Repetition penalty — reduce probability of tokens already generated
                for token_id in set(generated[0].tolist()):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, top_k)
                    threshold = top_k_vals[0, -1]
                    next_logits[next_logits < threshold] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cumprobs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = False
                    remove_indices = remove_mask.scatter(1, sorted_idx, remove_mask)
                    next_logits[remove_indices] = float("-inf")

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # EOS check
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, f"{path}/model.pt")

    @classmethod
    def load(cls, path: str, device="cpu"):
        ckpt = torch.load(f"{path}/model.pt", map_location=device, weights_only=False)
        config = ckpt["config"]
        model = cls(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model


# ============================================================
# 8. QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Sanskrit LGM — Architecture Test")
    print("=" * 60)

    configs = {
        "tiny (~1M)": SanskritConfig.tiny(),
        "small (~7M)": SanskritConfig.small(),
        "medium (~25M)": SanskritConfig.medium(),
        "large (~50M)": SanskritConfig.large(),
    }

    for name, config in configs.items():
        model = SanskritLGM(config)
        total, trainable = model.count_params()
        print(f"\n{name}:")
        print(f"  Parameters: {total:,}")
        print(f"  Layers: {config.num_layers}, Heads: {config.num_heads}")
        print(f"  Hidden: {config.hidden_dim}, FFN: {config.intermediate_dim}")

        # Quick forward pass test
        dummy = torch.randint(0, config.vocab_size, (1, 32))
        logits, _ = model(dummy)
        print(f"  Forward pass: input {dummy.shape} → logits {logits.shape} ✓")

        # Quick generate test
        start = torch.randint(0, config.vocab_size, (1, 5))
        gen = model.generate(start, max_new_tokens=10, eos_token_id=None)
        print(f"  Generate: {start.shape} → {gen.shape} ✓")

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)