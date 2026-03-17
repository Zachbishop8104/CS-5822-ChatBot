import torch
import numpy as np
from pathlib import Path

TOKENS_DIR = Path(__file__).parent.parent / "tokens"

def load_tokens(split="train"):
    bin_files = list(TOKENS_DIR.rglob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {TOKENS_DIR}")
    
    all_tokens = []
    for file in bin_files:
        tokens = np.fromfile(file, dtype=np.uint16).astype(np.int64)
        all_tokens.append(tokens)
        
    return np.concatenate(all_tokens)

tokens = load_tokens()
print(f"Total tokens loaded: {len(tokens):,}")

def get_batch(tokens, batch_size=8, seq_len=128):
    # pick random starting positions
    indices = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    
    # create input and target batches
    # the target is the goal, since it is shifted by one token, our model will learn to predict the next token
    input_batch = torch.stack([torch.from_numpy(tokens[i:i+seq_len].copy()) for i in indices])
    target_batch = torch.stack([torch.from_numpy(tokens[i+1:i+seq_len+1].copy()) for i in indices])
    
    return input_batch, target_batch

# split is typically 90% train, 10% validation
# will help use monitor the model's performance on unseen data during training and prevent overfitting
def split_tokens(tokens, val_ratio=0.1):
    split = int(len(tokens) * (1 - val_ratio))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]
    return train_tokens, val_tokens