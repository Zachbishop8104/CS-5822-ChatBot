import torch
import numpy as np
from pathlib import Path

TOKENS_DIR = Path(__file__).parent.parent / "tokens"

def load_tokens(max_tokens=None):
    bin_files = list(TOKENS_DIR.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {TOKENS_DIR}")

    # sort by size descending so large files load first
    bin_files = sorted(bin_files, key=lambda f: f.stat().st_size, reverse=True)

    all_tokens = []
    loaded = 0
    for file in bin_files:
        mm = np.memmap(file, dtype=np.uint16, mode="r")
        if max_tokens is not None:
            remaining = max_tokens - loaded
            if remaining <= 0:
                break
            mm = mm[:remaining]
        all_tokens.append(np.asarray(mm))
        loaded += len(mm)

    tokens = np.concatenate(all_tokens)
    print(f"Total tokens loaded: {len(tokens):,}")
    return tokens

def get_batch(tokens, batch_size=8, seq_len=128):
    indices = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    
    x = torch.empty((batch_size, seq_len), dtype=torch.long)
    y = torch.empty((batch_size, seq_len), dtype=torch.long)

    for i, idx in enumerate(indices):
        chunk = tokens[idx:idx+seq_len+1]
        x[i] = torch.from_numpy(chunk[:-1])
        y[i] = torch.from_numpy(chunk[1:])

    return x.pin_memory(), y.pin_memory()

# split is typically 90% train, 10% validation
# will help use monitor the model's performance on unseen data during training and prevent overfitting
def split_tokens(tokens, val_ratio=0.1):
    split = int(len(tokens) * (1 - val_ratio))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]
    return train_tokens, val_tokens