# reads data, trains model
# 1. Grab a batch of tokens from the .bin files
# 2. Feed them into the model -> get predictions
# 3. Compare predictions to the actual next tokens -> calculate loss
# 4. Backpropagate the loss -> update model weights
# 5. Repeat
import argparse
import torch
import torch.nn as nn
from model import Model
from dataloader import load_tokens, get_batch, split_tokens
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model_state"

def evaluate(model, val_tokens, loss_fn, device, vocab_size, steps=20):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(steps):
            input_batch, targets = get_batch(val_tokens)
            input_batch = input_batch.to(device)
            targets = targets.to(device)
            logits = model(input_batch)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / steps

def train_model():
    vocab_size = 16000
    model = Model(vocab_size=16000, embed_dim=512, num_heads=8, num_layers=6, dropout=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50_000)
    tokens = load_tokens(max_tokens=args.max_tokens)
    train_tokens, val_tokens = split_tokens(tokens)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    steps = 50_000

    for step in range(steps):
        input_batch, targets = get_batch(train_tokens, batch_size=64, seq_len=512)
        input_batch = input_batch.to(device)
        targets = targets.to(device)

        logits = model(input_batch)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            val_loss = evaluate(model, val_tokens, loss_fn, device, vocab_size)
            print(f"Step {step:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
        
        # save the model every 1000 steps
        if step % 1000 == 0 and step > 0:
            # saving by the class name in case we want to build more models with diferent versions 
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                MODEL_DIR / f"{model.__class__.__name__}.pth"
            )
    
    # Just in case we stop training before the next 1000 step checkpoint, save the latest model state
    if steps % 1000 != 0:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, MODEL_DIR / f"{model.__class__.__name__}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-training of the model")
    parser.add_argument("--model", type=str, default="Model", help="Model class name to train")
    parser.add_argument("--max_tokens", type=int, default=50_000_000, help="Maximum number of tokens to load into RAM")
    
    args = parser.parse_args()
    
    # train the model if they force it or if the model state doesn't exist yet
    if args.force or not (MODEL_DIR / f"{args.model}.pth").exists():
        train_model()
    else:
        print(f"Model state already exists at {MODEL_DIR / f'{args.model}.pth'}. Use --force to re-train.")
