# So at this point the model has learned how to predict the next token in general
# but it has no idea that it is a chat bot or how to answer questions.
# This file will finetune the model on a smaller dataset of Q&A pairs to teach it how to answer questions.

import torch
import torch.nn as nn
from model import Model
from tokenizer import load
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).parent.parent / "model_state"
QA_PATH = Path(__file__).parent.parent / "raw_text" / "qa" / "squad.txt"
TOKENS_DIR = Path(__file__).parent.parent / "tokens"

def load_qa_tokens():
    tok = load()
    tokens = []

    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.extend(tok.encode(line).ids)

    return np.array(tokens, dtype=np.uint16)

def get_batch(tokens, batch_size=32, seq_len=256):
    indices = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    input_batch  = torch.stack([torch.from_numpy(tokens[i:i+seq_len].astype(np.int64))     for i in indices])
    target_batch = torch.stack([torch.from_numpy(tokens[i+1:i+seq_len+1].astype(np.int64)) for i in indices])
    return input_batch, target_batch

def finetune():
    vocab_size = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained model
    model = Model(vocab_size, embed_dim=256, num_heads=4, num_layers=4, dropout=0.3)
    checkpoint = torch.load(MODEL_DIR / "Model.pth")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # lower learning rate for fine-tuning so we don't overwrite what was learned
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    print("Loading Q&A tokens...")
    tokens = load_qa_tokens()
    print(f"Q&A tokens: {len(tokens):,}")

    for step in range(3_000):
        input_batch, targets = get_batch(tokens)
        input_batch = input_batch.to(device)
        targets = targets.to(device)

        logits = model(input_batch)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
            
        # stop early if loss gets too low. Overfitting things
        if loss.item() < 0.5:
            print(f"Early stop at step {step} — loss {loss.item():.4f}")
            break

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        MODEL_DIR / "Model_finetuned.pth"
    )
    print("Saved finetuned model as Model_finetuned.pth")

if __name__ == "__main__":
    finetune()