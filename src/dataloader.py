import numpy as np
import torch
from pathlib import Path

TOKENS_DIR = Path(__file__).parent.parent / "tokens"


def _load_memmaps(val_ratio=0.1):
    """
    Load all .bin files and return train/val memmap slices.
    Splits within each file so val is always proportional to data size.
    """
    files = sorted(TOKENS_DIR.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"No .bin files found in {TOKENS_DIR}")

    train_memmaps, val_memmaps = [], []
    train_sizes,   val_sizes   = [], []

    for f in files:
        mm = np.memmap(f, dtype=np.uint16, mode="r")
        split = int(len(mm) * (1 - val_ratio))

        if split > 512:
            train_memmaps.append(mm[:split])
            train_sizes.append(split)

        if len(mm) - split > 512:
            val_memmaps.append(mm[split:])
            val_sizes.append(len(mm) - split)

    return (train_memmaps, np.array(train_sizes, dtype=np.float64),
            val_memmaps,   np.array(val_sizes,   dtype=np.float64))


# cache memmaps so we don't re-open files on every call
_cache = None

def _get_cache():
    global _cache
    if _cache is None:
        _cache = _load_memmaps()
    return _cache


def _sample(memmaps, sizes, seq_len):
    """Pick a random sequence from a weighted random file."""
    probs      = sizes / sizes.sum()
    idx        = np.random.choice(len(memmaps), p=probs)
    mm         = memmaps[idx]
    start      = np.random.randint(0, len(mm) - seq_len - 1)
    chunk      = mm[start:start + seq_len + 1].copy()
    x = torch.from_numpy(chunk[:-1].astype(np.int64))
    y = torch.from_numpy(chunk[1:].astype(np.int64))
    return x, y


def batch_generator(batch_size=32, seq_len=512, split="train"):
    """
    Infinite generator yielding (input, target) batches.

    Args:
        batch_size: number of sequences per batch
        seq_len:    tokens per sequence
        split:      "train" or "val"
    """
    train_mm, train_sz, val_mm, val_sz = _get_cache()
    memmaps = train_mm if split == "train" else val_mm
    sizes = train_sz if split == "train" else val_sz

    while True:
        xs, ys = [], []
        for _ in range(batch_size):
            x, y = _sample(memmaps, sizes, seq_len)
            xs.append(x)
            ys.append(y)
        yield torch.stack(xs).pin_memory(), torch.stack(ys).pin_memory()