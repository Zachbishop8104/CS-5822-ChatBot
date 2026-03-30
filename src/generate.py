import torch
import torch.nn.functional as F
from model import Model
from tokenizer import load
from pathlib import Path
import argparse
import re


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


def _tokenize_terms(text: str):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _is_numeric_request(prompt: str, instruction: str | None):
    combined = f"{prompt} {instruction or ''}".lower()
    hints = [
        "how many", "how much", "temperature", "percent", "rate", "number", "amount",
        "with units", "include both", "value", "values", "numeric",
    ]
    return any(h in combined for h in hints)


def _extractive_context_answer(
    prompt: str,
    instruction: str | None,
    context: str | None,
    max_sentences: int = 2,
):
    if not context or not context.strip():
        return None

    sentences = _split_sentences(context)
    if not sentences:
        return None

    query_terms = set(_tokenize_terms(prompt))
    query_terms.update(_tokenize_terms(instruction or ""))
    prefer_numeric = _is_numeric_request(prompt, instruction)

    scored = []
    for idx, sent in enumerate(sentences):
        sent_terms = set(_tokenize_terms(sent))
        overlap = len(query_terms & sent_terms)
        numeric_bonus = 2 if (prefer_numeric and re.search(r"\d", sent)) else 0
        score = overlap + numeric_bonus
        if score > 0:
            scored.append((score, idx, sent))

    if not scored:
        # Fallback to first sentence when no lexical overlap is found.
        return sentences[0]

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = sorted(scored[:max_sentences], key=lambda x: x[1])
    answer = " ".join(s for _, _, s in selected).strip()
    return answer


def _count_numeric_spans(text: str):
    return len(_extract_normalized_numbers(text))


def _extract_normalized_numbers(text: str):
    matches = re.findall(
        r"\d[\d,\.\s]*(?:\s*(?:million|billion|thousand))?(?:\s*°?\s*[CFK]|\s*%)?",
        text,
        flags=re.IGNORECASE,
    )
    normalized = []
    for m in matches:
        n = m.strip().lower()
        if not n:
            continue
        n = n.replace("°", "")
        n = re.sub(r"\s+", "", n)
        # Keep commas to preserve number identity (e.g., 5,500 vs 5500 in text forms).
        normalized.append(n)
    return normalized


def _should_auto_ground(
    prompt: str,
    instruction: str | None,
    context: str | None,
    answer: str,
):
    if not context or not context.strip():
        return False

    prefer_numeric = _is_numeric_request(prompt, instruction)
    if not prefer_numeric:
        return False

    context_nums = set(_extract_normalized_numbers(context))
    answer_nums = set(_extract_normalized_numbers(answer))

    if not answer_nums:
        return True

    # If generated numbers are not backed by context, fall back to grounded extraction.
    if context_nums and not answer_nums.issubset(context_nums):
        return True

    wants_both = bool(instruction and "both" in instruction.lower())
    if wants_both:
        context_count = len(context_nums)
        answer_count = len(answer_nums)
        if context_count >= 2 and answer_count < 2:
            return True

    return False

def _format_qa_prompt(
    prompt: str,
    context: str | None = None,
    instruction: str | None = None,
) -> str:
    p = prompt.strip()
    if "Question:" in p and "Answer:" in p:
        if instruction and "Instruction:" not in p:
            return p.replace("Answer:", f"Instruction: {instruction.strip()}\nAnswer:")
        return p

    lines = []
    if context and context.strip():
        lines.append(f"Context: {context.strip()}")
    lines.append(f"Question: {p}")
    if instruction and instruction.strip():
        lines.append(f"Instruction: {instruction.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def generate(
    prompt,
    max_new_tokens=64,
    temperature=0.2,
    top_k=20,
    prepend_bos=True,
    model_file_name="Model_finetuned_best.pth",
    repetition_penalty=1.15,
    context=None,
    instruction=None,
    strict_context_grounding=False,
    grounding_sentences=2,
    auto_grounding=False,
    return_source=False,
):
    source = "model"
    if strict_context_grounding:
        fallback = _extractive_context_answer(
            prompt=prompt,
            instruction=instruction,
            context=context,
            max_sentences=max(1, int(grounding_sentences)),
        )
        if fallback:
            source = "grounded-strict"
            if return_source:
                return fallback, source
            return fallback

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
    formatted_prompt = _format_qa_prompt(prompt, context=context, instruction=instruction)
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

    if auto_grounding and _should_auto_ground(prompt, instruction, context, decoded):
        fallback = _extractive_context_answer(
            prompt=prompt,
            instruction=instruction,
            context=context,
            max_sentences=max(1, int(grounding_sentences)),
        )
        if fallback:
            source = "grounded-auto"
            if return_source:
                return fallback, source
            return fallback

    if return_source:
        return decoded, source
    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file_name",
        default="Model_finetuned_best.pth",
        help="Checkpoint filename in model_state/ to load for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for greedy decoding; higher is more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k token filtering for sampling (set <=0 to disable).",
    )
    parser.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Prepend [BOS] token to the prompt before generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Penalty factor for previously seen tokens (1.0 disables penalty).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context passage used in prompt formatting and grounding.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional answer-style instruction included before Answer:.",
    )
    parser.add_argument(
        "--strict_context_grounding",
        action="store_true",
        help="Bypass model decoding and answer directly from context sentence extraction.",
    )
    parser.add_argument(
        "--grounding_sentences",
        type=int,
        default=2,
        help="Number of top context sentences to include when grounding is used.",
    )
    parser.add_argument(
        "--auto_grounding",
        action="store_true",
        help="Use model decoding first, then fallback to grounded extraction if unreliable.",
    )
    parser.add_argument(
        "--debug_grounding",
        action="store_true",
        help="Print answer source tag: model, grounded-strict, or grounded-auto.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text for non-interactive mode (skips Enter a prompt input).",
    )
    args = parser.parse_args()

    prompt = args.prompt if args.prompt is not None else input("Enter a prompt: ")
    output = generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        prepend_bos=args.prepend_bos,
        model_file_name=args.model_file_name,
        repetition_penalty=args.repetition_penalty,
        context=args.context,
        instruction=args.instruction,
        strict_context_grounding=args.strict_context_grounding,
        grounding_sentences=args.grounding_sentences,
        auto_grounding=args.auto_grounding,
        return_source=args.debug_grounding,
    )
    if args.debug_grounding:
        answer, source = output
        print(f"\n[source={source}]\n{answer}")
    else:
        print(f"\n{output}")