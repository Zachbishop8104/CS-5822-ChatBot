import argparse
import math
import torch
import torch.nn as nn
from model import Model
from dataloader import batch_generator
from pathlib import Path
from torch import amp
from tokenizer import load
from torch.optim.lr_scheduler import LambdaLR

MODEL_DIR = Path(__file__).parent.parent / "model_state"

STEPS = 100_000
BATCH_SIZE = 32
SEQ_LEN = 384
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
EVAL_INTERVAL = 500
SAVE_INTERVAL = 10_000
WARMUP_STEPS = 2_000

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

def evaluate(model, val_stream, loss_fn, device, vocab_size, steps=20):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(steps):
            input_batch, targets = next(val_stream)
            input_batch = input_batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(input_batch)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / steps

def train_model():
    tok = load()
    vocab_size = tok.get_vocab_size()
    model = Model(vocab_size=vocab_size, embed_dim=768, num_heads=12, num_layers=8, dropout=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = amp.GradScaler()

    train_stream = batch_generator(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, split="train")
    val_stream = batch_generator(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, split="val")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    total_tokens = STEPS * tokens_per_step
    print(
        f"Training config | steps={STEPS:,} batch_size={BATCH_SIZE} seq_len={SEQ_LEN} "
        f"lr={LEARNING_RATE:.2e} weight_decay={WEIGHT_DECAY} "
        f"tokens/step={tokens_per_step:,} total_tokens~={total_tokens:,}"
    )

    best_val_loss = float('inf')
    best_step = 0

    for step in range(STEPS):
        input_batch, targets = next(train_stream)
        input_batch = input_batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp.autocast(device_type=device.type):
            logits = model(input_batch)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % EVAL_INTERVAL == 0:
            train_loss = evaluate(model, train_stream, loss_fn, device, vocab_size)
            val_loss = evaluate(model, val_stream, loss_fn, device, vocab_size)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step:5d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            
            # Track and save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                    },
                    MODEL_DIR / f"{model.__class__.__name__}_best.pth"
                )
                print(f"*New best val loss: {val_loss:.4f} @ step {step}")

        if step % SAVE_INTERVAL == 0 and step > 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                },
                MODEL_DIR / f"{model.__class__.__name__}_ckpt_{step}.pth"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": STEPS,
        },
        MODEL_DIR / f"{model.__class__.__name__}_final.pth"
    )
    print(f"\nTraining complete.")
    print(f"Best model:   {model.__class__.__name__}_best.pth (step {best_step}, val_loss {best_val_loss:.4f})")
    print(f"Final model:  {model.__class__.__name__}_final.pth (step {STEPS})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-training of the model")
    parser.add_argument("--model", type=str, default="Model", help="Model class name to train")
    args = parser.parse_args()

    if args.force or not (MODEL_DIR / f"{args.model}.pth").exists():
        train_model()
    else:
        print(f"Model state already exists at {MODEL_DIR / f'{args.model}.pth'}. Use --force to re-train.")