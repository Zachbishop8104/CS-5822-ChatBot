# finetune.py
# The model only learns to predict the answer tokens — prompt tokens are masked.

import torch
import torch.nn as nn
from model import Model
from tokenizer import load
from pathlib import Path
from finetune_get_data import load_all_qa_blocks
import numpy as np
import argparse
from torch import amp
from torch.optim.lr_scheduler import LambdaLR
import math

MODEL_DIR = Path(__file__).parent.parent / "model_state"


def _make_lr_lambda(steps: int, warmup: int):
    warmup = max(1, warmup)
    steps  = max(2, steps)
    def _fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return _fn

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _normalize_state_dict(state_dict: dict) -> dict:
    """Strip torch.compile '_orig_mod.' prefix if present."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qa_pairs(val_ratio: float = 0.05):
    """
    Parse [NOTE_QA] blocks into (prompt, answer) pairs.
    Handles two completion styles:
      "Answer: {text}"
      "Explanation: {text}"
    prompt = everything up to and including the completion token
    answer = the text that follows it
    """
    examples = load_all_qa_blocks()
    if len(examples) < 2:
        raise ValueError(
            f"Only {len(examples)} QA blocks found. "
            "Run `python finetune_get_data.py` first to download data."
        )

    # Ordered longest-first so rfind picks the most specific match
    COMPLETION_TOKENS = [
        "Explanation:",
        "Answer:",
    ]

    pairs = []
    skipped = 0
    for block in examples:
        matched = False
        for token in COMPLETION_TOKENS:
            idx = block.rfind(token)
            if idx == -1:
                continue
            prompt = block[:idx + len(token)].strip()
            answer = block[idx + len(token):].strip()
            answer = answer.split("\n\n")[0].strip()
            if prompt and answer:
                pairs.append((prompt, answer))
                matched = True
            break
        if not matched:
            skipped += 1

    if skipped:
        print(f"  (skipped {skipped} malformed blocks)")

    rng = np.random.default_rng(42)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]

    split = max(1, min(int(len(pairs) * (1 - val_ratio)), len(pairs) - 1))
    return pairs[:split], pairs[split:], len(pairs)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def get_batch(pairs, tok, batch_size: int, seq_len: int, bos_id, eos_id):
    """
    Sample batch_size (prompt, answer) pairs and encode them.
    Returns:
        input_ids : (batch, seq_len)  long tensor
        mask      : (batch, seq_len)  bool tensor — True where loss should fire (answer tokens)
    """
    pad_id = tok.token_to_id("[PAD]") or 0
    indices = np.random.choice(len(pairs), batch_size, replace=len(pairs) < batch_size)

    input_batch, mask_batch = [], []
    for idx in indices:
        prompt, answer = pairs[idx]

        prompt_ids = tok.encode(prompt).ids
        answer_ids = tok.encode(answer).ids

        if bos_id is not None:
            prompt_ids = [bos_id] + prompt_ids
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        # Truncate: preserve as much answer as possible
        total = len(prompt_ids) + len(answer_ids)
        if total > seq_len:
            budget = seq_len - len(answer_ids)
            if budget > 0:
                # Keep the start (Context: [text]) AND the end (... Question: [q] Explanation:)
                # We reserve the last 40 tokens for the question/trigger, and use the rest for context.
                tail_len = min(40, budget)
                head_len = budget - tail_len
                prompt_ids = prompt_ids[:head_len] + prompt_ids[-tail_len:]
            else:
                # Fallback if the answer alone exceeds seq_len (rare with your dataset)
                prompt_ids = []
                answer_ids = answer_ids[:seq_len]

        ids  = prompt_ids + answer_ids
        mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

        # Pad
        pad = seq_len - len(ids)
        if pad > 0:
            ids  += [pad_id] * pad
            mask += [0] * pad

        input_batch.append(ids)
        mask_batch.append(mask)

    return (
        torch.tensor(input_batch, dtype=torch.long),
        torch.tensor(mask_batch,  dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(
    model_file_name: str,
    steps: int       = 9_000,
    batch_size: int  = 32,
    seq_len: int     = 512,
    lr: float        = 1e-5,
    eval_interval: int       = 200,
    val_ratio: float         = 0.05,
    early_stop_patience: int = 100,
):
    tok        = load()
    vocab_size = tok.get_vocab_size()
    bos_id     = tok.token_to_id("[BOS]")
    eos_id     = tok.token_to_id("[EOS]")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load checkpoint --
    ckpt_path = MODEL_DIR / model_file_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state       = _normalize_state_dict(checkpoint["model"])

    emb_key = "embedding.embedding.weight"
    if emb_key not in state:
        raise KeyError(f"Missing key '{emb_key}' in checkpoint after normalisation.")
    if int(state[emb_key].shape[0]) != vocab_size:
        raise ValueError(
            f"Vocab mismatch: tokenizer={vocab_size}, "
            f"checkpoint={state[emb_key].shape[0]}"
        )

    model = Model(
        vocab_size=vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=8,
        dropout=0.1,
    )
    model.load_state_dict(state)
    model.to(device)

    # -- Optimiser & scheduler --
    WARMUP    = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, _make_lr_lambda(steps, WARMUP))
    loss_fn   = nn.CrossEntropyLoss()
    scaler    = amp.GradScaler(enabled=(device.type == "cuda"))

    # -- Data --
    print("Loading QA pairs...")
    train_pairs, val_pairs, n_total = load_qa_pairs(val_ratio)
    print(
        f"  Total: {n_total:,}  |  train: {len(train_pairs):,}  |  val: {len(val_pairs):,}"
    )
    if not train_pairs or not val_pairs:
        raise ValueError("Not enough QA pairs for train/val split.")

    print(
        f"\nFine-tune config | model={model_file_name}  steps={steps:,}  "
        f"batch={batch_size}  seq_len={seq_len}  lr={lr:.2e}  device={device}"
    )

    # -- Output path --
    def unique_path(base: str) -> Path:
        p = MODEL_DIR / base
        if not p.exists():
            return p
        stem, ext = base.rsplit(".", 1) if "." in base else (base, "")
        i = 1
        while True:
            candidate = MODEL_DIR / (f"{stem}_{i}.{ext}" if ext else f"{stem}_{i}")
            if not candidate.exists():
                return candidate
            i += 1

    out_path       = unique_path("Model_finetuned_best.pth")
    best_val_loss  = float("inf")
    best_step      = -1
    bad_eval_count = 0

    # -- Training loop --
    for step in range(steps):
        model.train()
        x, mask = get_batch(train_pairs, tok, batch_size, seq_len, bos_id, eos_id)
        x    = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with amp.autocast(device_type=device.type):
            logits = model(x)                          # (B, T, V)
            # Causal shift: predict token t+1 from token t
            logits_s = logits[:, :-1, :].contiguous()  # (B, T-1, V)
            targets  = x[:, 1:].contiguous()           # (B, T-1)
            mask_s   = mask[:, 1:].contiguous()        # (B, T-1)

            # Only backprop through answer tokens
            active_logits  = logits_s.view(-1, vocab_size)[mask_s.view(-1)]
            active_targets = targets.view(-1)[mask_s.view(-1)]

            if active_targets.numel() == 0:
                continue  # skip degenerate batch (shouldn't happen)

            loss = loss_fn(active_logits, active_targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # -- Validation --
        if step % eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    vx, vmask = get_batch(val_pairs, tok, batch_size, seq_len, bos_id, eos_id)
                    vx    = vx.to(device, non_blocking=True)
                    vmask = vmask.to(device, non_blocking=True)

                    with amp.autocast(device_type=device.type):
                        vlogits  = model(vx)[:, :-1, :].contiguous()
                        vtargets = vx[:, 1:].contiguous()
                        vmask_s  = vmask[:, 1:].contiguous()

                        al = vlogits.view(-1, vocab_size)[vmask_s.view(-1)]
                        at = vtargets.view(-1)[vmask_s.view(-1)]
                        if at.numel() > 0:
                            val_losses.append(loss_fn(al, at).item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            print(
                f"Step {step:5d} | train_loss: {loss.item():.4f} | val_loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_step      = step
                bad_eval_count = 0
                torch.save(
                    {
                        "model":      model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "base_model": model_file_name,
                        "step":       step,
                        "batch_size": batch_size,
                        "seq_len":    seq_len,
                        "lr":         lr,
                        "val_loss":   val_loss,
                    },
                    out_path,
                )
                print(f"  * New best val_loss {val_loss:.4f} @ step {step}  →  {out_path.name}")
            else:
                bad_eval_count += 1
                if early_stop_patience > 0 and bad_eval_count >= early_stop_patience:
                    print(
                        f"Early stop: no improvement for {bad_eval_count} evals "
                        f"({bad_eval_count * eval_interval} steps)."
                    )
                    break

    if best_step < 0:
        raise RuntimeError("No validation step ran — no model was saved.")

    print(
        f"\nDone. Best model: {out_path.name}  "
        f"(step {best_step}, val_loss {best_val_loss:.4f})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_name",      default="Model_best.pth")
    parser.add_argument("--steps",                type=int,   default=9_000)
    parser.add_argument("--batch_size",           type=int,   default=32)
    parser.add_argument("--seq_len",              type=int,   default=256)
    parser.add_argument("--lr",                   type=float, default=1e-5)
    parser.add_argument("--eval_interval",        type=int,   default=200)
    parser.add_argument("--val_ratio",            type=float, default=0.05)
    parser.add_argument("--early_stop_patience",  type=int,   default=100)
    args = parser.parse_args()

    finetune(
        model_file_name     = args.model_file_name,
        steps               = args.steps,
        batch_size          = args.batch_size,
        seq_len             = args.seq_len,
        lr                  = args.lr,
        eval_interval       = args.eval_interval,
        val_ratio           = args.val_ratio,
        early_stop_patience = args.early_stop_patience,
    )