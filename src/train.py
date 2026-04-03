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
BATCH_SIZE = 24
SEQ_LEN = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
EVAL_INTERVAL = 500
WARMUP_STEPS = 2_000
GRAD_ACCUM_STEPS = 3


def _checkpoint_paths(model_name):
    return {
        "best": MODEL_DIR / f"{model_name}_best.pth",
        "final": MODEL_DIR / f"{model_name}_final.pth",
    }


def _infer_max_seq_len_from_checkpoint(checkpoint_path, default_max_seq_len):
    if not checkpoint_path.exists():
        return default_max_seq_len

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pos_embed_weight = checkpoint["model"].get("positional_encoding.pos_embed.weight")
    if pos_embed_weight is None:
        return default_max_seq_len
    return max(default_max_seq_len, int(pos_embed_weight.shape[0]))

def make_lr_lambda(steps, warmup_steps):
    warmup_steps = max(1, int(warmup_steps))
    steps = max(2, int(steps))

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, (steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return _lr_lambda

def lr_lambda(step):
    # Backward-compatible default schedule using module constants.
    return make_lr_lambda(STEPS, WARMUP_STEPS)(step)

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


def _load_checkpoint_if_available(model, optimizer, scheduler, model_name):
    checkpoint_paths = _checkpoint_paths(model_name)
    checkpoint_path = checkpoint_paths["best"] if checkpoint_paths["best"].exists() else checkpoint_paths["final"]

    if not checkpoint_path.exists():
        return 0, float("inf"), None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_step = int(checkpoint.get("step", -1)) + 1
    best_val_loss = float(checkpoint.get("val_loss", float("inf")))
    best_step = int(checkpoint.get("best_step", checkpoint.get("step", -1)))
    return start_step, best_val_loss, best_step


def train_model(
    steps=STEPS,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_interval=EVAL_INTERVAL,
    warmup_steps=WARMUP_STEPS,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    resume=True,
    model_name="Model",
):

    tok = load()
    vocab_size = tok.get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_stream = batch_generator(batch_size=batch_size, seq_len=seq_len, split="train")
    val_stream = batch_generator(batch_size=batch_size, seq_len=seq_len, split="val")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    grad_accum_steps = max(1, int(grad_accum_steps))
    tokens_per_step = batch_size * seq_len * grad_accum_steps
    total_tokens = steps * tokens_per_step
    print(
        f"Training config | steps={steps:,} batch_size={batch_size} seq_len={seq_len} "
        f"grad_accum_steps={grad_accum_steps} "
        f"lr={learning_rate:.2e} weight_decay={weight_decay} "
        f"tokens/step={tokens_per_step:,} total_tokens~={total_tokens:,}"
    )

    checkpoint_paths = _checkpoint_paths(model_name)
    if resume:
        candidate_path = checkpoint_paths["best"] if checkpoint_paths["best"].exists() else checkpoint_paths["final"]
        max_seq_len = _infer_max_seq_len_from_checkpoint(candidate_path, max(seq_len, 1024))
    else:
        max_seq_len = max(seq_len, 1024)

    model = Model(
        vocab_size=vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=8,
        dropout=0.1,
        max_seq_len=max_seq_len,
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=(device.type == "cuda"))
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, make_lr_lambda(steps, warmup_steps))
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    if resume:
        try:
            start_step, best_val_loss, best_step = _load_checkpoint_if_available(model, optimizer, scheduler, model_name)
        except RuntimeError as error:
            print(f"Checkpoint resume failed for {model_name}: {error}")
            print("Starting fresh because the checkpoint architecture no longer matches this model.")
            start_step, best_val_loss, best_step = 0, float("inf"), 0
    else:
        start_step, best_val_loss, best_step = 0, float("inf"), 0

    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    if resume and start_step > 0:
        print(
            f"Resuming from step {start_step - 1} using {model_name}_"
            f"{'best' if (MODEL_DIR / f'{model_name}_best.pth').exists() else 'final'}.pth"
        )

    if start_step >= steps:
        print(f"Requested steps ({steps}) are not greater than checkpoint step ({start_step - 1}); nothing to train.")
        return

    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum_steps):
            input_batch, targets = next(train_stream)
            input_batch = input_batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(input_batch)
                loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % eval_interval == 0:
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
                        "best_step": best_step,
                        "val_loss": val_loss,
                    },
                    MODEL_DIR / f"{model_name}_best.pth"
                )
                print(f"*New best val loss: {val_loss:.4f} @ step {step}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": steps - 1,
            "best_step": best_step,
            "val_loss": best_val_loss,
        },
        MODEL_DIR / f"{model_name}_final.pth"
    )
    print(f"\nTraining complete.")
    print(f"Best model:   {model_name}_best.pth (step {best_step}, val_loss {best_val_loss:.4f})")
    print(f"Final model:  {model_name}_final.pth (step {steps})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-training of the model")
    parser.add_argument("--model", type=str, default="Model", help="Model class name to train")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    args = parser.parse_args()

    if args.force:
        train_model(
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            warmup_steps=args.warmup_steps,
            grad_accum_steps=args.grad_accum_steps,
            resume=False,
            model_name=args.model,
        )
    else:
        train_model(
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            warmup_steps=args.warmup_steps,
            grad_accum_steps=args.grad_accum_steps,
            resume=True,
            model_name=args.model,
        )