import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from wikitext.bpe_tokenizer import BPETokenizer


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    tokenizer: str
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    architecture: str
    positional: str


MODEL_SPECS = {
    "v1_1_char_scaled": ModelSpec(
        name="v1_1_char_scaled",
        checkpoint="v1_models/best_model_v1_1.pth",
        tokenizer="char",
        block_size=256,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=0.2,
        architecture="legacy_mha",
        positional="learned",
    ),
    "v1_2_alibi_char": ModelSpec(
        name="v1_2_alibi_char",
        checkpoint="v1_models/best_model.pth",
        tokenizer="char",
        block_size=512,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=0.2,
        architecture="projected_mha",
        positional="alibi",
    ),
    "v2_2_rope_bpe": ModelSpec(
        name="v2_2_rope_bpe",
        checkpoint="v2/v2_2/best_model.pth",
        tokenizer="Vocabs/bpe_vocab.json",
        block_size=256,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=0.2,
        architecture="projected_mha",
        positional="rope",
    ),
    "v2_3_rope_bpe_scaled": ModelSpec(
        name="v2_3_rope_bpe_scaled",
        checkpoint="v2/v2_3/best_model.pth",
        tokenizer="Vocabs/bpe_vocab.json",
        block_size=512,
        n_embd=384,
        n_head=8,
        n_layer=6,
        dropout=0.25,
        architecture="projected_mha",
        positional="rope",
    ),
    "v2_4_rope_bpe_500_scaled": ModelSpec(
        name="v2_4_rope_bpe_500_scaled",
        checkpoint="best_model.pth",
        tokenizer="Vocabs/bpe_vocab_500.json",
        block_size=256,
        n_embd=256,
        n_head=6,
        n_layer=6,
        dropout=0.1,
        architecture="projected_mha",
        positional="rope",
    ),
    "v3_0_rope_english_bpe": ModelSpec(
        name="v3_0_rope_english_bpe",
        checkpoint="v3/v3_0/best_modelv3_0.pth",
        tokenizer="Vocabs/bpe_vocab_english_500.json",
        block_size=256,
        n_embd=256,
        n_head=6,
        n_layer=6,
        dropout=0.1,
        architecture="projected_mha",
        positional="rope",
    ),
    "v3_1_alibi_english_bpe": ModelSpec(
        name="v3_1_alibi_english_bpe",
        checkpoint="v3/v3_1/best_modelv3_1.pth",
        tokenizer="Vocabs/bpe_vocab_english_500.json",
        block_size=512,
        n_embd=256,
        n_head=6,
        n_layer=6,
        dropout=0.2,
        architecture="projected_mha",
        positional="alibi",
    ),
}


class CharTokenizer:
    def __init__(self, input_path):
        text = Path(input_path).read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab = self.stoi

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)


def load_tokenizer(spec):
    if spec.tokenizer == "char":
        return CharTokenizer(REPO_DIR / "input.txt")
    tokenizer = BPETokenizer()
    tokenizer.load(REPO_DIR / spec.tokenizer)
    return tokenizer


def get_alibi_slopes(n_heads):
    def get_slopes_power_of_2(n):
        start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    return torch.tensor(
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
            : n_heads - closest_power_of_2
        ]
    )


def apply_rope(x):
    bsz, n_heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    freqs = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
            / head_dim
        )
    )
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    return rotated


class LegacyHead(nn.Module):
    def __init__(self, spec, head_size):
        super().__init__()
        self.key = nn.Linear(spec.n_embd, head_size, bias=False)
        self.query = nn.Linear(spec.n_embd, head_size, bias=False)
        self.value = nn.Linear(spec.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(spec.block_size, spec.block_size)))
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, x):
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class LegacyMultiHeadAttention(nn.Module):
    def __init__(self, spec, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([LegacyHead(spec, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, spec.n_embd)
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class ProjectedMultiHeadAttention(nn.Module):
    def __init__(self, spec, num_heads, head_size):
        super().__init__()
        self.spec = spec
        self.num_heads = num_heads
        self.head_size = head_size
        self.q_proj = nn.Linear(spec.n_embd, num_heads * head_size, bias=False)
        self.k_proj = nn.Linear(spec.n_embd, num_heads * head_size, bias=False)
        self.v_proj = nn.Linear(spec.n_embd, num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, spec.n_embd)
        self.dropout = nn.Dropout(spec.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(spec.block_size, spec.block_size)))
        if spec.positional == "alibi":
            self.register_buffer("alibi_slopes", get_alibi_slopes(num_heads), persistent=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        if self.spec.positional == "rope":
            q = apply_rope(q)
            k = apply_rope(k)

        wei = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        if self.spec.positional == "alibi":
            pos = torch.arange(seq_len, device=x.device)
            bias = pos.unsqueeze(0) - pos.unsqueeze(1)
            bias = bias.unsqueeze(0).unsqueeze(0) * self.alibi_slopes.view(1, -1, 1, 1)
            wei = wei + bias

        wei = wei.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_size)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, spec):
        super().__init__()
        if spec.architecture == "legacy_mha":
            self.net = nn.Sequential(
                nn.Linear(spec.n_embd, 4 * spec.n_embd),
                nn.ReLU(),
                nn.Linear(4 * spec.n_embd, spec.n_embd),
                nn.Dropout(spec.dropout),
            )
        else:
            self.proj_in = nn.Linear(spec.n_embd, 4 * spec.n_embd * 2)
            self.proj_out = nn.Linear(4 * spec.n_embd, spec.n_embd)
            self.dropout = nn.Dropout(spec.dropout)

    def forward(self, x):
        if hasattr(self, "net"):
            return self.net(x)
        x1, x2 = self.proj_in(x).chunk(2, dim=-1)
        return self.dropout(self.proj_out(F.gelu(x1) * x2))


class Block(nn.Module):
    def __init__(self, spec):
        super().__init__()
        head_size = spec.n_embd // spec.n_head
        if spec.architecture == "legacy_mha":
            self.sa = LegacyMultiHeadAttention(spec, spec.n_head, head_size)
        else:
            self.sa = ProjectedMultiHeadAttention(spec, spec.n_head, head_size)
        self.ffwd = FeedForward(spec)
        self.ln1 = nn.LayerNorm(spec.n_embd)
        self.ln2 = nn.LayerNorm(spec.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, spec, vocab_size):
        super().__init__()
        self.spec = spec
        self.token_embedding_table = nn.Embedding(vocab_size, spec.n_embd)
        if spec.positional == "learned":
            self.position_embedding_table = nn.Embedding(spec.block_size, spec.n_embd)
        self.blocks = nn.Sequential(*[Block(spec) for _ in range(spec.n_layer)])
        self.ln_f = nn.LayerNorm(spec.n_embd)
        if spec.architecture == "legacy_mha":
            self.lm_head = nn.Linear(spec.n_embd, vocab_size)
        else:
            self.llm_head = nn.Linear(spec.n_embd, vocab_size)
            self.llm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape
        x = self.token_embedding_table(idx)
        if hasattr(self, "position_embedding_table"):
            positions = torch.arange(seq_len, device=idx.device)
            x = x + self.position_embedding_table(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) if hasattr(self, "lm_head") else self.llm_head(x)

        loss = None
        if targets is not None:
            bsz, steps, channels = logits.shape
            loss = F.cross_entropy(logits.view(bsz * steps, channels), targets.view(bsz * steps))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.spec.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_model(spec_name, device="cpu"):
    spec = MODEL_SPECS[spec_name]
    tokenizer = load_tokenizer(spec)
    vocab_size = len(tokenizer.vocab)
    model = LanguageModel(spec, vocab_size).to(device)
    checkpoint = torch.load(REPO_DIR / spec.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = {
        "iter": checkpoint.get("iter"),
        "params": sum(p.numel() for p in model.parameters()),
        "spec": spec.__dict__,
    }
    return model, tokenizer, metadata


def write_registry(path):
    data = {name: spec.__dict__ for name, spec in MODEL_SPECS.items()}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
