import random

# There was a lot of effort done to get this data, holy
def process_eli5_jsonl_kaggle(
    input_path=None,
    output_path=None,
    n_samples=20000,
    min_words=10,
    max_words=40,
    seed=42
):
    """
    Process a large ELI5.jsonl file from Kaggle, randomly sample up to n_samples with answer length filtering
    """
    import json
    from pathlib import Path
    import random
    input_path = input_path or (Path(__file__).parent.parent / "ELI5.jsonl")
    output_path = output_path or (Path(__file__).parent.parent / "raw_text" / "qa" / "eli5_noteqa.txt")
    random.seed(seed)
    print(f"Processing ELI5 Kaggle: {input_path} -> {output_path}")
    # First pass: collect line offsets for eligible examples
    offsets = []
    with open(input_path, "r", encoding="utf-8") as fin:
        pos = 0
        while True:
            line = fin.readline()
            if not line:
                break
            try:
                ex = json.loads(line)
            except Exception:
                pos += len(line)
                continue
            question = (ex.get("question") or "").strip()
            ctxs = ex.get("ctxs", [])
            context = ctxs[0].strip() if ctxs and isinstance(ctxs, list) and ctxs[0] else ""
            answers = ex.get("answers", [])
            answer = answers[0].strip() if answers and isinstance(answers, list) and answers[0] else ""
            if not (question and context and answer):
                pos += len(line)
                continue
            n_words = len(answer.split())
            if n_words < min_words or n_words > max_words:
                pos += len(line)
                continue
            offsets.append(pos)
            pos += len(line)
    print(f"Eligible ELI5 examples: {len(offsets):,}")
    if len(offsets) == 0:
        print("No eligible ELI5 examples found.")
        return
    sample_offsets = random.sample(offsets, min(n_samples, len(offsets)))
    sample_offsets_set = set(sample_offsets)
    # Second pass: write sampled examples
    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        pos = 0
        while True:
            line = fin.readline()
            if not line:
                break
            if pos in sample_offsets_set:
                try:
                    ex = json.loads(line)
                except Exception:
                    pos += len(line)
                    continue
                question = (ex.get("question") or "").strip()
                ctxs = ex.get("ctxs", [])
                context = ctxs[0].strip() if ctxs and isinstance(ctxs, list) and ctxs[0] else ""
                answers = ex.get("answers", [])
                answer = answers[0].strip() if answers and isinstance(answers, list) and answers[0] else ""
                if not (question and context and answer):
                    pos += len(line)
                    continue
                n_words = len(answer.split())
                if n_words < min_words or n_words > max_words:
                    pos += len(line)
                    continue
                fout.write(f"[Context: {context}\nQuestion: {question}\nExplanation: {answer}\n\n")
                written += 1
                if written % 1000 == 0:
                    print(f"  Wrote {written:,} examples...")
                if written >= n_samples:
                    break
            pos += len(line)
    print(f"Done. Wrote {written:,} ELI5 examples to {output_path}")
if __name__ == "__main__":
    process_eli5_jsonl_kaggle()