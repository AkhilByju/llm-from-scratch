import os
import torch 
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from wikitext.bpe_tokenizer import BPETokenizer
from torchtune.modules import RotaryPositionalEmbeddings


# hyperparameters
batch_size = 32
block_size = 512
max_iters = 9000
eval_interval = 300 
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2
# --------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

torch.manual_seed(1337)

# Get the dataset
with open(os.path.join(BASE_DIR, "input.txt"), "r", encoding="utf-8") as f:
    data = f.read()

# Character Tokenization
bpe = BPETokenizer()
bpe.load(os.path.join(BASE_DIR, "bpe_vocab.json"))

vocab_size = len(bpe.vocab)

train_cache = "train_ids.pt"
val_cache = "val_ids.pt"

if os.path.exists(os.path.join(BASE_DIR, train_cache)) and os.path.exists(os.path.join(BASE_DIR, val_cache)):
    print("Loading pre-encoded dataset...")
    train_data = torch.load(os.path.join(BASE_DIR, train_cache))
    val_data = torch.load(os.path.join(BASE_DIR, val_cache))
else:
    print("Encoding dataset...")
    ids = bpe.encode(data, show_progress=True)
    print("Enconding done, length: ", len(ids))
    decode = bpe.decode

    # Train-Validation split
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:] 

    # Save
    torch.save(train_data,  train_cache)
    torch.save(val_data, val_cache)
    print("Saved encoded dataset to disk.")

encode = bpe.encode
decode = bpe.decode

# load in the data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        return q, k, v

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.q_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbeddings(dim=head_size, max_seq_len=block_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)  # (B, n_h, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # apply RoPE
        q = self.rope(q, input_pos=None)
        k = self.rope(k, input_pos=None)

        # attention scores
        wei = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))  # (B, n_h, T, T)

        # causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted sum
        out = wei @ v  # (B, n_h, T, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.num_heads * self.head_size)  # back to (B, T, C)

        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, expansion=4):
        super().__init__()
        self.proj_in = nn.Linear(n_embd, expansion * n_embd * 2)  # double for split
        self.proj_out = nn.Linear(expansion * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        hidden = self.proj_in(x)                       # (B, T, 2*expansion*n_embd)
        x1, x2 = hidden.chunk(2, dim=-1)               # split into two
        x = F.gelu(x1) * x2                            # gated GELU
        x = self.proj_out(x)                           # back to (B, T, n_embd)
        return self.dropout(x)

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.llm_head = nn.Linear(n_embd, vocab_size)

        self.llm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.llm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

model = LanguageModel(vocab_size).to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# setup for saving best model
best_val_loss = float("inf")

checkpoint_path = "checkpoint.pth"
start_iter = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iter"]
    print(f"✅ Resumed from iteration {start_iter}")

for iter in range(start_iter, max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        # Save checkpoint
        torch.save({
            "iter": iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"💾 Saved checkpoint at step {iter}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                "iter": iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, "best_model.pth")
            print(f"🌟 New best model saved at step {iter} (val loss {best_val_loss:.4f})")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({
    "iter": max_iters,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, checkpoint_path)
print("✅ Final checkpoint saved.")

# generate from the model
start_text = input("Enter a starting phrase: ")

if start_text.strip() == "":
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
else:
    context = torch.tensor([encode(start_text)], dtype=torch.long, device=device)

print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))