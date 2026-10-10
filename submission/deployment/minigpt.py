"""Standalone MiniGPT inference module (model + character tokenizer + checkpoint loader).

Mirrors the architecture trained in the notebook: token embedding + sinusoidal positions,
N pre-norm causal Transformer blocks, final LayerNorm, linear LM head.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(T, device=None):
    """(T, T) boolean mask, True where the key position is in the future of the query position."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def scaled_dot_product_attention(Q, K, V, causal=True):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if causal:
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        mask = build_causal_mask(max(T_q, T_k), device=scores.device)[:T_q, :T_k]
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x):
        Q, K, V = self.split_heads(self.W_q(x)), self.split_heads(self.W_k(x)), self.split_heads(self.W_v(x))
        out, attn = scaled_dot_product_attention(Q, K, V, causal=self.causal)
        return self.W_o(self.merge_heads(out)), attn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.0, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn = self.attn(self.ln1(x))
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x, attn


@dataclass
class GPTConfig:
    vocab_size: int = 65
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = None
    dropout: float = 0.1
    tie_weights: bool = False

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.d_model


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.context_length)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        assert idx.shape[1] <= self.cfg.context_length
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)
        x = self.drop(self.pos_enc(x))
        for block in self.blocks:
            x, _ = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.context_length:])[:, -1, :]
            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s, errors="raise"):
        ids = []
        for c in s:
            if c in self.stoi:
                ids.append(self.stoi[c])
            elif errors != "skip":
                raise KeyError(f"character {c!r} is not in the vocabulary")
        return ids

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_model(path, device="cpu"):
    """Load a checkpoint written by the notebook. Returns (model, tokenizer, checkpoint_dict)."""
    device = torch.device(device)
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["gpt_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["tokenizer_chars"]), ckpt


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=None):
    """Prompt string in -> prompt + continuation string out."""
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, errors="skip") or tokenizer.encode("\n")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
