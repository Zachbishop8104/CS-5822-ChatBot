import json
import random
from pathlib import Path

# Data Augmentation: Mimic Slide Notes
def _corrupt_for_slides(text: str, corruption_chance: float = 0.5) -> str:
    """Randomly formats pristine text to look like messy slide notes."""
    if random.random() > corruption_chance:
        return text 

    words = text.split()
    corrupted = []
    
    for i, word in enumerate(words):
        # Strip punctuation randomly (simulating shorthand)
        if random.random() < 0.5:
            word = word.replace(".", "").replace(",", "").replace(";", "")
            
        corrupted.append(word)
        
        # Randomly break lines mid-sentence (simulating slide text wrap)
        if random.random() < 0.15 and i < len(words) - 1:
            corrupted.append("\n")

    # Join back together and clean up spacing around the newlines
    result = " ".join(corrupted)
    result = result.replace(" \n ", "\n").replace("\n - ", "\n- ")
    return result.strip()


def _truncate(text: str, max_words: int = 120) -> str:
    """Keep contexts from blowing out the sequence length."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def process_eli5_jsonl_kaggle(
    input_path=None,
    output_path=None,
    n_samples=20000,
    min_words=10,
    max_words=40,
    seed=42
):
    """
    Process a large ELI5.jsonl file from Kaggle, randomly sample up to n_samples 
    with answer length filtering, and apply slide-note corruption.
    """
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
                    
                # Apply data augmentation and truncation
                context = _corrupt_for_slides(context)
                context = _truncate(context, max_words=120)
                
                # Removed the stray '[' bracket that was here previously!
                fout.write(f"Context: {context}\nQuestion: {question}\nExplanation: {answer}\n\n")
                written += 1
                
                if written % 1000 == 0:
                    print(f"  Wrote {written:,} examples...")
                if written >= n_samples:
                    break
            pos += len(line)
            
    print(f"Done. Wrote {written:,} ELI5 examples to {output_path}")

if __name__ == "__main__":
    process_eli5_jsonl_kaggle()