from datasets import load_dataset
from pathlib import Path
from tokenizer import load

QA_DIR = Path(__file__).parent.parent / "raw_text" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

# List of QA datasets: (dataset_name, config, split, output_file, n_samples)
SOURCES = [
    ("squad_v2", None, "train", "squad.txt", 12_000),
    ("openbookqa", "main", "train", "openbookqa.txt", 5_000),
    ("blended_skill_talk", None, "train", "blended_skill_talk.txt", 12_000),
    ("trivia_qa", "rc", "train", "triviaqa.txt", 12_000),
    ("natural_questions", "default", "train", "natural_questions.txt", 12_000),
    ("boolq", None, "train", "boolq.txt", 12_000),
    ("ai2_arc", "ARC-Challenge", "train", "arc_challenge.txt", 1_119),
    ("ai2_arc", "ARC-Easy", "train", "arc_easy.txt", 2_251),
    ("pubmed_qa", "pqa_labeled", "train", "pubmed_qa.txt", 1_000),
]

ACTIVE_QA_FILES = [out_file for _, _, _, out_file, _ in SOURCES]

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
    elif dataset == "trivia_qa":
        question = ex.get("question", "").strip()
        answer = ex.get("answer", {}).get("value", "").strip()
        context = ""
        return context, question, answer
    elif dataset == "natural_questions":
        question = ex.get("question", {}).get("text", "").strip()
        annotations = ex.get("annotations", {}).get("short_answers", [])
        text_list = annotations[0].get("text", []) if annotations else []
        answer = text_list[0] if text_list else ""
        context = ""
        return context, question, answer
    elif dataset == "boolq":
        question = ex.get("question", "").strip()
        context = ex.get("passage", "").strip()
        answer_val = ex.get("answer", None)
        if isinstance(answer_val, bool):
            answer = "yes" if answer_val else "no"
        else:
            answer = str(answer_val).strip() if answer_val is not None else ""
        return context, question, answer
    elif dataset == "ai2_arc":
        question = ex.get("question", "").strip()
        choices = ex.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        texts = choices.get("text", []) if isinstance(choices, dict) else []
        answer_key = str(ex.get("answerKey", "")).strip()
        answer = ""
        if answer_key in labels:
            idx = labels.index(answer_key)
            answer = texts[idx].strip() if idx < len(texts) else ""
        elif answer_key.isdigit():
            idx = int(answer_key) - 1
            if 0 <= idx < len(texts):
                answer = texts[idx].strip()
        context = ""
        return context, question, answer
    elif dataset == "pubmed_qa":
        question = ex.get("question", "").strip()
        context_obj = ex.get("context", {})
        contexts = context_obj.get("contexts", []) if isinstance(context_obj, dict) else []
        context = " ".join(c.strip() for c in contexts if isinstance(c, str) and c.strip())
        answer = str(ex.get("final_decision", "")).strip().lower()
        return context, question, answer
    elif dataset == "eli5":
        question = ex.get("title", "").strip()
        context = ex.get("selftext", "").strip()
        if not question and context:
            question, context = context, ""
        answers = ex.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer_scores = answers.get("score", []) if isinstance(answers, dict) else []
        if answer_texts:
            if answer_scores and len(answer_scores) == len(answer_texts):
                best_idx = max(range(len(answer_scores)), key=answer_scores.__getitem__)
                answer = answer_texts[best_idx].strip()
            else:
                answer = answer_texts[0].strip()
        else:
            answer = ""
        return context, question, answer
    elif dataset == "cosmos_qa":
        context = ex.get("context", "").strip()
        question = ex.get("question", "").strip()
        choices = [ex.get("answer0", ""), ex.get("answer1", ""), ex.get("answer2", ""), ex.get("answer3", "")]
        label = ex.get("label", -1)
        answer = ""
        if isinstance(label, int) and 0 <= label < len(choices):
            answer = str(choices[label]).strip()
        return context, question, answer
    return "", "", ""


def dump_qa_bulk():
    tok = load()
    for name, config, split, out_file, n_samples in SOURCES:
        print(f"Processing {name}...")
        kwargs = {"streaming": True}
        if config:
            kwargs["name"] = config
        try:
            ds = load_dataset(name, split=split, **kwargs)
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                print(f"Skipping {name}: unsupported dataset script in installed datasets version")
                continue
            raise
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

def load_all_qa_blocks(source_files=None):
    """Return QA blocks from the active finetune source files."""
    blocks = []
    if source_files is None:
        source_files = ACTIVE_QA_FILES
    for file_name in source_files:
        file = QA_DIR / file_name
        if not file.exists():
            continue
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                file_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
                blocks.extend(file_blocks)
    return blocks

if __name__ == "__main__":
    dump_qa_bulk()