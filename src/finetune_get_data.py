from datasets import load_dataset
from pathlib import Path
from tokenizer import load

QA_DIR = Path(__file__).parent.parent / "raw_text" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

# List of QA datasets: (dataset_name, config, split, output_file, n_samples)
SOURCES = [
    ("squad_v2", None, "train", "squad.txt", 50_000),
    ("openbookqa", "main", "train", "openbookqa.txt", 20_000),
    ("blended_skill_talk", None, "train", "blended_skill_talk.txt", 20_000),
]

def extract_qa(ex, dataset):
    """Extract context, question, answer from a dataset example."""
    if dataset == "squad_v2":
        context = ex.get("context", "").strip()
        question = ex.get("question", "").strip()
        answers = ex.get("answers", {}).get("text", [])
        answer = answers[0] if answers else ""
        return context, question, answer
    elif dataset == "openbookqa":
        question = ex.get("question_stem", "").strip()
        choices = ex.get("choices", {}).get("text", [])
        answer_idx = ex.get("answerKey", "A")
        # OpenBookQA uses answerKey as letter, choices as list
        idx = ord(answer_idx) - ord("A")
        answer = choices[idx] if idx < len(choices) else ""
        context = ""
        return context, question, answer
    elif dataset == "blended_skill_talk":
        prev = ex.get("previous_utterance", "")
        if isinstance(prev, list):
            context = " ".join(p.strip() for p in prev if isinstance(p, str)).strip()
        elif isinstance(prev, str):
            context = prev.strip()
        else:
            context = ""
        free_msgs = ex.get("free_messages", [""])
        guided_msgs = ex.get("guided_messages", [""])
        question = free_msgs[0].strip() if free_msgs and isinstance(free_msgs[0], str) else ""
        answer = guided_msgs[0].strip() if guided_msgs and isinstance(guided_msgs[0], str) else ""
        return context, question, answer
    # Add more dataset-specific logic as needed
    return "", "", ""


def dump_qa_bulk():
    tok = load()
    for name, config, split, out_file, n_samples in SOURCES:
        print(f"Processing {name}...")
        kwargs = {"streaming": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(name, split=split, **kwargs)
        out_path = QA_DIR / out_file
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in ds:
                context, question, answer = extract_qa(ex, name)
                if question and answer:
                    line = (
                        (f"Context: {context}\n" if context else "") +
                        f"Question: {question}\n"
                        f"Answer: {answer}\n\n"
                    )
                    f.write(line)
                    count += 1
                if count >= n_samples:
                    break
        print(f"Saved {count} Q&A pairs -> {out_path}")

def load_all_qa_blocks():
    """Return a list of QA blocks (separated by blank lines) from all .txt files in QA_DIR."""
    blocks = []
    for file in sorted(QA_DIR.glob("*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                file_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
                blocks.extend(file_blocks)
    return blocks

if __name__ == "__main__":
    dump_qa_bulk()