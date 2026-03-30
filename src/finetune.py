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
from torch import amp

MODEL_DIR = Path(__file__).parent.parent / "model_state"
QA_PATH = Path(__file__).parent.parent / "raw_text" / "qa" / "squad.txt"

def _load_qa_examples():
    """Load QA blocks separated by blank lines, preserving prompt/answer format."""
    with open(QA_PATH, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def load_qa_tokens(val_ratio=0.05):
    tok = load()
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")

    examples = _load_qa_examples()
    if len(examples) < 2:
        raise ValueError("Need at least 2 QA examples for train/val split.")

    encoded = []
    for ex in examples:
        ids = tok.encode(ex).ids
        if bos_id is not None:
            ids = [bos_id] + ids
        if eos_id is not None:
            ids = ids + [eos_id]
        encoded.append(ids)

    split_idx = int(len(encoded) * (1 - val_ratio))
    split_idx = max(1, min(split_idx, len(encoded) - 1))

    train_tokens = [tid for sample in encoded[:split_idx] for tid in sample]
    val_tokens = [tid for sample in encoded[split_idx:] for tid in sample]

    return (
        np.array(train_tokens, dtype=np.uint16),
        np.array(val_tokens, dtype=np.uint16),
        len(examples),
    )

def get_batch(tokens, batch_size=32, seq_len=256):
    if len(tokens) <= seq_len + 1:
        raise ValueError(
            f"Token stream too short for seq_len={seq_len}. Got {len(tokens):,} tokens."
        )

    indices = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
    input_batch  = torch.stack([torch.from_numpy(tokens[i:i+seq_len].astype(np.int64))     for i in indices])
    target_batch = torch.stack([torch.from_numpy(tokens[i+1:i+seq_len+1].astype(np.int64)) for i in indices])
    return input_batch, target_batch


def evaluate(model, tokens, loss_fn, vocab_size, device, batch_size, seq_len, steps=20):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(steps):
            input_batch, targets = get_batch(tokens, batch_size=batch_size, seq_len=seq_len)
            input_batch = input_batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type):
                logits = model(input_batch)
                loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            total += loss.item()
    model.train()
    return total / steps


def finetune(model_file_name, steps=3000, batch_size=32, seq_len=256, lr=5e-6, eval_interval=200, val_ratio=0.05):
    tok = load()
    vocab_size = tok.get_vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_DIR / model_file_name, map_location="cpu", weights_only=True)
    ckpt_vocab_size = int(checkpoint["model"]["embedding.embedding.weight"].shape[0])
    if ckpt_vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({vocab_size}) does not match checkpoint vocab ({ckpt_vocab_size}) in {model_file_name}. "
            "Use a checkpoint trained with the current tokenizer or retrain first."
        )

    # must match the architecture of the pretrained model
    model = Model(vocab_size=vocab_size, embed_dim=768, num_heads=12, num_layers=8, dropout=0.1)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    print("Loading Q&A tokens...")
    train_tokens, val_tokens, n_examples = load_qa_tokens(val_ratio=val_ratio)
    print(
        f"Q&A examples: {n_examples:,} | train_tokens: {len(train_tokens):,} | "
        f"val_tokens: {len(val_tokens):,}"
    )

    if len(train_tokens) <= seq_len + 1 or len(val_tokens) <= seq_len + 1:
        raise ValueError(
            "QA token stream too short for current seq_len. "
            "Lower seq_len or increase QA data."
        )

    print(
        f"Fine-tune config | model={model_file_name} steps={steps:,} "
        f"batch_size={batch_size} seq_len={seq_len} lr={lr:.2e}"
    )

    best_val_loss = float("inf")
    best_step = -1
    best_output_name = "Model_finetuned_best.pth"

    for step in range(steps):
        input_batch, targets = get_batch(train_tokens, batch_size=batch_size, seq_len=seq_len)
        input_batch = input_batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp.autocast(device_type=device.type):
            logits = model(input_batch)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % eval_interval == 0:
            val_loss = evaluate(
                model,
                val_tokens,
                loss_fn,
                vocab_size,
                device,
                batch_size=batch_size,
                seq_len=seq_len,
                steps=20,
            )
            print(f"Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "base_model": model_file_name,
                        "steps": step,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "lr": lr,
                        "val_loss": val_loss,
                    },
                    MODEL_DIR / best_output_name,
                )
                print(f"*New best val loss: {val_loss:.4f} @ step {step}")
    if best_step >= 0:
        print(
            f"Best finetuned model: {best_output_name} "
            f"(step {best_step}, val_loss {best_val_loss:.4f})"
        )
    else:
        raise RuntimeError("No validation step ran; no finetuned model was saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_name", default="Model_best.pth")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    finetune(
        model_file_name=args.model_file_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        eval_interval=args.eval_interval,
        val_ratio=args.val_ratio,
    )
