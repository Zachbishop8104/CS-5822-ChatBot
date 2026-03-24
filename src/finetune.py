# So at this point the model has learned how to predict the next token in general
# but it has no idea that it is a chat bot or how to answer questions.
# This file will finetune the model on a smaller dataset of Q&A pairs to teach it how to answer questions.

import torch
import torch.nn as nn
from model import Model
from tokenizer import load
from pathlib import Path
import numpy as np
import argparse

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

def finetune(model_file_name):
    vocab_size = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # must match the architecture of the pretrained model
    model = Model(vocab_size=16000, embed_dim=768, num_heads=12, num_layers=8, dropout=0.1)
    checkpoint = torch.load(MODEL_DIR / model_file_name, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
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

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        MODEL_DIR / "Model_finetuned.pth"
    )
    print("Saved finetuned model as Model_finetuned.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_name")
    args = parser.parse_args()

    if args.model_file_name:
        finetune(args.model_file_name)
    else:
        print("No model file name provided. Use --model_file_name to specify the pretrained model to finetune.")
