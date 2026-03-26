import torch
import torch.nn.functional as F
from model import Model
from tokenizer import load
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model_state" / "Model_finetuned.pth"

def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=None):
    # Load tokenizer and model
    tok = load()
    vocab_size = tok.get_vocab_size()
    model = Model(vocab_size=vocab_size, embed_dim=768, num_heads=12, num_layers=8, dropout=0.1)

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    # Move to GPU and optimize memory for generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()   # disables dropout
    model.half()   # convert model to float16 for memory efficiency

    # Prepare input
    input_ids = torch.tensor([tok.encode(prompt).ids], device=device, dtype=torch.long)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids.half())  # forward pass in float16
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            # Optional top-k sampling for better generation
            if top_k is not None:
                top_vals, top_idx = torch.topk(probs, top_k)
                probs = F.softmax(top_vals, dim=-1)
                next_token = top_idx[0, torch.multinomial(probs, num_samples=1)]
                next_token = next_token.unsqueeze(0).unsqueeze(0)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tok.token_to_id("[EOS]"):
                break

    return tok.decode(input_ids[0].tolist())


if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    output = generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f"\n{output}")