import torch
import torch.nn.functional as F
from model import Model
from tokenizer import load
from pathlib import Path
import argparse
import re

def _format_qa_prompt(prompt: str, context: str | None = None) -> str:
    p = prompt.strip()
    if "Question:" in p and "Answer:" in p:
        return p
    if context and context.strip():
        return f"Context: {context.strip()}\nQuestion: {p}\nAnswer:"
    return f"Question: {p}\nAnswer:"


def generate(
    prompt,
    max_new_tokens=64,
    temperature=0.2,
    top_k=20,
    prepend_bos=True,
    model_file_name="Model_finetuned_best.pth",
    repetition_penalty=1.15,
    context=None,
):
    tok = load()
    vocab_size = tok.get_vocab_size()
    model_path = Path(__file__).parent.parent / "model_state" / model_file_name

    # Initialize model
    model = Model(vocab_size=vocab_size, embed_dim=768, num_heads=12, num_layers=8, dropout=0.1)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()

    # Prepare input IDs
    formatted_prompt = _format_qa_prompt(prompt, context=context)
    ids = tok.encode(formatted_prompt).ids
    if prepend_bos:
        bos_id = tok.token_to_id("[BOS]")
        if bos_id is not None:
            ids = [bos_id] + ids
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    eos_id = tok.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            if temperature <= 0:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                next_logits = logits[:, -1, :] / temperature

                # Penalize previously generated tokens to reduce loops.
                if repetition_penalty is not None and repetition_penalty > 1.0:
                    seen = torch.unique(input_ids)
                    next_logits[:, seen] = next_logits[:, seen] / repetition_penalty

                # Top-k sampling in logit space
                if top_k is not None and top_k > 0:
                    k = min(top_k, next_logits.size(-1))
                    top_vals, top_idx = torch.topk(next_logits, k, dim=-1)
                    top_probs = F.softmax(top_vals, dim=-1)
                    choice = torch.multinomial(top_probs, num_samples=1)
                    next_token = top_idx.gather(-1, choice)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            # Ensure 2D for concatenation
            next_token = next_token.to(torch.long)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if eos_id is not None and next_token.item() == eos_id:
                break

            # Stop if the model starts a new QA turn.
            partial = tok.decode(input_ids[0].tolist())
            question_count = len(re.findall(r"Question\s*:", partial))
            if question_count > 1:
                break

    # Decode, strip BOS/EOS
    decoded = tok.decode(input_ids[0].tolist())
    decoded = decoded.replace("[BOS]", "").replace("[EOS]", "").strip()

    # Return only the current answer span.
    decoded = re.split(r"Answer\s*:", decoded, maxsplit=1)
    decoded = decoded[1].strip() if len(decoded) > 1 else decoded[0].strip()
    decoded = re.split(r"Question\s*:", decoded, maxsplit=1)[0].strip()
    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_name", default="Model_finetuned_best.pth")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--prepend_bos", action="store_true")
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--context", type=str, default=None)
    args = parser.parse_args()

    prompt = input("Enter a prompt: ")
    output = generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        prepend_bos=args.prepend_bos,
        model_file_name=args.model_file_name,
        repetition_penalty=args.repetition_penalty,
        context=args.context,
    )
    print(f"\n{output}")