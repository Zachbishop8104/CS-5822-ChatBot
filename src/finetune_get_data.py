from datasets import load_dataset
from pathlib import Path
from tokenizer import load

QA_DIR = Path(__file__).parent.parent / "raw_text" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

def dump_qa(n_samples=50_000):
    tok = load()
    ds = load_dataset("squad_v2", split="train", streaming=True)

    out_path = QA_DIR / "squad.txt"
    count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            context = ex.get("context", "").strip()
            question = ex.get("question", "").strip()
            answers  = ex.get("answers", {}).get("text", [])
            if context and question and answers:
                line = (
                    f"Context: {context}\n"
                    f"Question: {question}\n"
                    f"Answer: {answers[0]}\n\n"
                )
                f.write(line)
                count += 1
            if count >= n_samples:
                break

    print(f"Saved {count} Q&A pairs → {out_path}")

if __name__ == "__main__":
    dump_qa()