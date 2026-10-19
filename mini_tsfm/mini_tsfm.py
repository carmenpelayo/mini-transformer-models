"""Mini-TSFM: a tiny decoder-only time-series foundation-model teaching example.

Install: pip install torch matplotlib
Run:     python mini_tsfm.py --steps 300
Quick:   python mini_tsfm.py --steps 3 --eval-interval 1

A TSFM is made 'foundation-like' by pretraining on MANY distinct time series.
This example uses synthetic series so it runs without an external dataset.
It forecasts real values (regression), not discrete vocabulary IDs.
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 1. Hyperparameters: keep the model small enough to understand and run on CPU.
# -----------------------------------------------------------------------------
batch_size = 32               # independent windows per optimization step
context_patches = 16         # maximum number of input patch tokens
patch_size = 8               # raw time steps in each token (16*8 = 128 steps)
d_model = 64                 # embedding dimension (the GPT 'n_embd')
n_head = 4                   # attention heads (64 / 4 = 16 dimensions/head)
n_layer = 2                  # Transformer blocks
dropout = 0.1
learning_rate = 3e-4
seed = 1337
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 2. Data: multiple distinct series, to mimic broad TSFM pretraining.
#    Substitute this function with real historical series for genuine pretraining.
#    Splits are by SERIES, not just overlapping windows of the same series.
# -----------------------------------------------------------------------------
def make_series(n_series, length, generator):
    """Synthesize signals with varied trend, cycles, persistence and noise."""
    t = torch.arange(length, dtype=torch.float32)
    result = []
    for _ in range(n_series):
        period1 = 14 + 85 * torch.rand((), generator=generator).item()
        period2 = 6 + 25 * torch.rand((), generator=generator).item()
        phase = 2 * math.pi * torch.rand((), generator=generator).item()
        amp = 0.4 + 2.0 * torch.rand((), generator=generator).item()
        slope = (torch.rand((), generator=generator).item() - 0.5) * 0.012
        level = 3 * (torch.rand((), generator=generator).item() - 0.5)
        noise = torch.randn(length, generator=generator) * 0.12
        # AR(1) noise adds temporal dependence beyond fixed seasonal patterns.
        ar = torch.zeros(length)
        for j in range(1, length):
            ar[j] = 0.75 * ar[j - 1] + noise[j]
        y = (level + slope * t
             + amp * torch.sin(2 * math.pi * t / period1 + phase)
             + 0.25 * amp * torch.sin(2 * math.pi * t / period2)
             + ar)
        result.append(y)
    return torch.stack(result)  # (number_of_series, time_steps)

# -----------------------------------------------------------------------------
# 3. Patch tokenization: continuous-valued counterpart of GPT char tokenization.
#    Characters -> token IDs -> lookup embeddings in GPT.
#    Values -> fixed-length patches -> LINEAR projection in mini-TSFM.
#    No discrete vocabulary and NO rounding/quantization are needed.
# -----------------------------------------------------------------------------
def patchify(x):
    """(B, T) -> (B, T/patch_size, patch_size), with T divisible by patch_size."""
    B, T = x.shape
    assert T % patch_size == 0
    return x.reshape(B, T // patch_size, patch_size)


def normalize_with_context(values, context_length):
    """Instance normalization using ONLY observed context, not future targets.

    The same mean/std transform applies to context and targets. Return stats
    so future forecasts can be converted back to the original numeric scale.
    """
    observed = values[:, :context_length]
    mu = observed.mean(dim=1, keepdim=True)
    sigma = observed.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (values - mu) / sigma, mu, sigma


def get_batch(data, generator=None):
    """One observed context plus the *next* patch for next-patch supervision.

    Raw window: (B, (context_patches + 1)*patch_size).
    Input tokens: patches[:, :-1]                  (B, P, S)
    Labels:       patches[:, 1:]                   (B, P, S)
    Each token predicts the FOLLOWING patch, analogous to next-token GPT.
    """
    needed = (context_patches + 1) * patch_size
    assert data.shape[1] >= needed
    series_idx = torch.randint(data.shape[0], (batch_size,), generator=generator)
    starts = torch.randint(data.shape[1] - needed + 1, (batch_size,), generator=generator)
    offsets = torch.arange(needed)
    windows = data[series_idx[:, None], starts[:, None] + offsets[None, :]]
    windows = windows.to(DEVICE)
    z, _, _ = normalize_with_context(windows, context_patches * patch_size)
    patches = patchify(z)
    return patches[:, :-1], patches[:, 1:]

# -----------------------------------------------------------------------------
# 4. Causal multi-head attention: identical core idea to Karpathy's mini-GPT.
#    Token i can attend to tokens <= i, never to future tokens.
# -----------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_patches, context_patches)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        attention = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention = self.dropout(F.softmax(attention, dim=-1))
        return attention @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert d_model % n_head == 0
        head_size = d_model // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # residual attention
        x = x + self.ff(self.ln2(x))  # residual MLP
        return x

# -----------------------------------------------------------------------------
# 5. The model: linear PATCH embedding -> position -> Transformer -> PATCH head.
#    Last output token predicts the NEXT patch of real-valued observations.
# -----------------------------------------------------------------------------
class MiniTSFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.position_embedding = nn.Embedding(context_patches, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.forecast_head = nn.Linear(d_model, patch_size)

    def forward(self, x, targets=None):
        """x: (B, P, patch_size), targets: following patches (same shape)."""
        B, P, S = x.shape
        assert S == patch_size and P <= context_patches
        positions = torch.arange(P, device=x.device)
        h = self.patch_embedding(x) + self.position_embedding(positions)
        h = self.ln_f(self.blocks(h))
        pred = self.forecast_head(h)          # (B, P, patch_size), no softmax
        loss = F.mse_loss(pred, targets) if targets is not None else None
        return pred, loss

    @torch.no_grad()
    def forecast(self, history, horizon):
        """Forecast horizon raw time steps, rolling one PATCH at a time.

        history: (B, T), where T is a multiple of patch_size.
        Normalize using history ONLY. Keep at most context_patches tokens.
        Recursively append the last predicted patch, then denormalize.
        """
        assert history.ndim == 2 and history.shape[1] >= patch_size
        assert history.shape[1] % patch_size == 0 and horizon >= 1
        history = history.to(next(self.parameters()).device)
        z, mu, sigma = normalize_with_context(history, history.shape[1])
        tokens = patchify(z)[:, -context_patches:, :]
        predicted = []
        for _ in range(math.ceil(horizon / patch_size)):
            next_patch = self(tokens)[0][:, -1:, :]  # (B, 1, patch_size)
            predicted.append(next_patch)
            tokens = torch.cat([tokens, next_patch], dim=1)[:, -context_patches:, :]
        scaled = torch.cat(predicted, dim=1).reshape(history.shape[0], -1)[:, :horizon]
        return scaled * sigma + mu

# -----------------------------------------------------------------------------
# 6. Optimization: same optimizer loop as mini-GPT; MSE replaces cross-entropy.
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=10):
    model.eval()
    results = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--out', type=str, default='mini_tsfm_forecast.png')
    args = parser.parse_args()

    torch.manual_seed(seed)
    random.seed(seed)
    # Different series in train and validation, so validation tests transfer.
    train_data = make_series(160, 512, torch.Generator().manual_seed(10))
    val_data = make_series(32, 512, torch.Generator().manual_seed(20))
    model = MiniTSFM().to(DEVICE)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'device={DEVICE} | parameters={n_parameters:,} | ' 
          f'context={context_patches * patch_size} steps | patch={patch_size}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(args.steps):
        if step % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters=5)
            print(f"step {step:4d}: train MSE {losses['train']:.4f} | "
                  f"validation MSE {losses['val']:.4f}")
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # -------------------------------------------------------------------------
    # 7. Zero-shot example: unseen synthetic series, no additional fine-tuning.
    # -------------------------------------------------------------------------
    model.eval()
    demo = make_series(1, 200, torch.Generator().manual_seed(999)).to(DEVICE)
    observed = demo[:, :128]
    forecast = model.forecast(observed, horizon=48)  # 48 next raw observations
    print('First 8 forecast values:', forecast[0, :8].cpu().tolist())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(128), observed[0].cpu(), label='Observed context', lw=2)
    ax.plot(range(128, 176), demo[0, 128:176].cpu(), label='Actual future', ls='--')
    ax.plot(range(128, 176), forecast[0].cpu(), label='Mini-TSFM forecast', lw=2)
    ax.axvline(127.5, color='gray', alpha=0.6)
    ax.set(xlabel='Time step', ylabel='Value', title='Forecast on an unseen series')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f'Wrote {Path(args.out).resolve()}')

if __name__ == '__main__':
    main()
