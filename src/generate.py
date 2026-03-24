import torch
from model import Model
from tokenizer import load
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model_state" / "Model_finetuned.pth"

def generate(prompt, max_new_tokens=100, temperature=0.8):
    tok = load()
    model = Model(vocab_size=16000, embed_dim=768, num_heads=12, num_layers=8, dropout=0.1)
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_ids = torch.tensor([tok.encode(prompt).ids])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_logit = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logit, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tok.decode(input_ids[0].tolist())

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    response = generate(prompt)
    print(f"\n{response}")