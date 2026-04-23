from retrieve import format_prompt
import torch
import torch.nn.functional as F
from pathlib import Path
from model import Model
from tokenizer import load

MODEL_DIR = Path(__file__).parent.parent / "model_state"
DEFAULT_CHECKPOINT = "Model_finetuned_best.pth"


def load_model(checkpoint_name=DEFAULT_CHECKPOINT, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = load()
    vocab_size = tok.get_vocab_size()

    checkpoint = torch.load(MODEL_DIR / checkpoint_name, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    pos_key = "positional_encoding.pos_embed.weight"
    max_seq_len = int(state[pos_key].shape[0]) if pos_key in state else 1024

    model = Model(
        vocab_size=vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=8,
        dropout=0.0,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tok, device


@torch.no_grad()
def generate(
    model,
    tok,
    device,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.3,
):
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")

    encoded = tok.encode(prompt).ids
    if bos_id is not None:
        encoded = [bos_id] + encoded

    prompt_len = len(encoded) 

    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    generated = list(encoded)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.positional_encoding.pos_embed.weight.shape[0]:]

        logits = model(context)
        next_logits = logits[0, -1, :].float()

        if repetition_penalty != 1.0:
            for token_id in set(generated[prompt_len:]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        next_logits = next_logits / max(temperature, 1e-8)

        if top_k > 0:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < values[-1]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            next_logits = torch.zeros_like(next_logits).scatter_(
                0, sorted_indices, sorted_logits
            )

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if eos_id is not None and next_token == eos_id:
            break

        generated.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=device)], dim=1
        )

    output_ids = generated[1:] if (bos_id is not None and generated[0] == bos_id) else generated
    return tok.decode(generated[prompt_len:])


def chat(checkpoint_name=DEFAULT_CHECKPOINT, username: str = None):
    print(f"Loading model from {checkpoint_name}...")
    model, tok, device = load_model(checkpoint_name)
    print(f"Model loaded on {device}. Type 'quit' to exit.\n")

    if not username:
        username = input("Username: ").strip()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        prompt = format_prompt(username, user_input)
        response = generate(
            model, tok, device,
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.3,
        )
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--username", type=str, default=None, help="Username (notes folder)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.username:
        print("Error: --username is required", file=sys.stderr)
        sys.exit(1)

    if args.prompt:
        t0 = time.time()
        model, tok, device = load_model(args.checkpoint)
        t1 = time.time()

        prompt = format_prompt(args.username, args.prompt)
        response = generate(
            model, tok, device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        t2 = time.time()

        print(response)

        if args.debug:
            print(f"[DEBUG] Username:         {args.username}",          file=sys.stderr)
            print(f"[DEBUG] Question:         {args.prompt}",            file=sys.stderr)
            print(f"[DEBUG] Prompt sent:      {prompt}",                 file=sys.stderr)
            print(f"[DEBUG] Model:            {args.checkpoint}",        file=sys.stderr)
            print(f"[DEBUG] Load time:        {t1-t0:.3f}s",             file=sys.stderr)
            print(f"[DEBUG] Generation time:  {t2-t1:.3f}s",             file=sys.stderr)
    else:
        chat(args.checkpoint, username=args.username)