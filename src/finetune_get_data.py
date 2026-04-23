from datasets import load_dataset
from pathlib import Path
import re
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

QA_DIR = Path(__file__).parent.parent / "raw_text" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

MIN_ANSWER_WORDS  = 5
MIN_CONTEXT_WORDS = 20
MAX_ANSWER_WORDS  = 40   # cap ELI5 answers so they don't ramble

# (hf_dataset_name, config_or_None, split, output_file, n_samples, completion_style)
# completion_style: "answer" -> "Answer:" | "explain" -> "Explanation:"
SOURCES = [
    ("squad_v2",      None,          "train", "squad.txt",        15_000, "answer"),
    ("narrativeqa",   None,          "train", "narrativeqa.txt",   8_000, "answer"),
    ("newsqa",        None,          "train", "newsqa.txt",        8_000, "answer"),
    ("hotpot_qa",     "distractor",  "train", "hotpotqa.txt",     10_000, "answer"),
    ("pubmed_qa",     "pqa_labeled", "train", "pubmed_qa.txt",     1_000, "answer"),
    ("ms_marco",      "v1.1",        "train", "ms_marco.txt",     12_000, "answer"),
]

ACTIVE_QA_FILES = [out_file for _, _, _, out_file, _, _ in SOURCES]
ACTIVE_QA_FILES.append("eli5_noteqa.txt")  # Kaggle ELI5 processed separately


# Helpers

def _clean(text) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, max_words: int = 120) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _trim_answer(answer: str, max_words: int = MAX_ANSWER_WORDS) -> str:
    """Trim answer to max_words, ending at a sentence boundary if possible."""
    words = answer.split()
    if len(words) <= max_words:
        return answer
    trimmed = " ".join(words[:max_words])
    # Try to end at last sentence boundary
    last_period = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if last_period > len(trimmed) // 2:
        return trimmed[:last_period + 1]
    return trimmed + "."


# Per-dataset extractors  ->  (context, question, answer) or None

def _extract(ex, dataset: str):

    if dataset == "squad_v2":
        context  = _clean(ex.get("context", ""))
        question = _clean(ex.get("question", ""))
        answers  = ex.get("answers", {}).get("text", [])
        answer   = _clean(answers[0]) if answers else ""
        if not answer:
            return None
        return context, question, answer

    elif dataset == "narrativeqa":
        doc = ex.get("document", {})
        if isinstance(doc, dict):
            summary = doc.get("summary", {})
            context = _clean(
                summary.get("text", "") if isinstance(summary, dict) else str(summary)
            )
        else:
            context = ""
        question = _clean(
            ex.get("question", {}).get("text", "")
            if isinstance(ex.get("question"), dict) else ""
        )
        ans_list = ex.get("answers", [])
        answer   = _clean(
            ans_list[0].get("text", "")
            if ans_list and isinstance(ans_list[0], dict) else ""
        )
        if not answer:
            return None
        return context, question, answer

    elif dataset == "newsqa":
        context   = _clean(ex.get("story_text", ""))
        question  = _clean(ex.get("question", ""))
        answers   = ex.get("answers", {})
        ans_texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer    = _clean(ans_texts[0]) if ans_texts else ""
        if not answer:
            return None
        return context, question, answer

    elif dataset == "hotpot_qa":
        ctx_obj   = ex.get("context", {})
        sentences = ctx_obj.get("sentences", []) if isinstance(ctx_obj, dict) else []
        flat = []
        for sent_list in sentences:
            if isinstance(sent_list, list):
                flat.extend(sent_list)
            elif isinstance(sent_list, str):
                flat.append(sent_list)
        context  = _clean(" ".join(flat))
        question = _clean(ex.get("question", ""))
        answer   = _clean(ex.get("answer", ""))
        if not answer:
            return None
        return context, question, answer

    elif dataset == "pubmed_qa":
        question  = _clean(ex.get("question", ""))
        ctx_obj   = ex.get("context", {})
        ctx_parts = ctx_obj.get("contexts", []) if isinstance(ctx_obj, dict) else []
        context   = _clean(" ".join(c for c in ctx_parts if isinstance(c, str)))
        answer    = _clean(str(ex.get("final_decision", ""))).lower()
        expand    = {
            "yes":   "Yes, the evidence provided supports this conclusion.",
            "no":    "No, the evidence does not support this conclusion.",
            "maybe": "The evidence is mixed and a definitive answer is unclear.",
        }
        answer = expand.get(answer, answer)
        return context, question, answer

    elif dataset == "ms_marco":
        passages_obj  = ex.get("passages", {})
        passage_texts = (
            passages_obj.get("passage_text", [])
            if isinstance(passages_obj, dict) else []
        )
        context  = _clean(passage_texts[0]) if passage_texts else ""
        question = _clean(ex.get("query", ""))
        answers  = ex.get("answers", [])
        answer   = _clean(answers[0]) if answers else ""
        if not answer or answer.lower() in {"no answer present.", "no answer present"}:
            return None
        return context, question, answer

    elif dataset == "eli5_category":
        question   = _clean(ex.get("title", ""))
        # Use the selftext as context if available, otherwise first document
        context    = _clean(ex.get("selftext", ""))
        if not context or len(context.split()) < MIN_CONTEXT_WORDS:
            docs    = ex.get("documents", [])
            context = _clean(docs[0]) if docs else ""
        # Pick highest scored answer
        answers    = ex.get("answers", {})
        ans_texts  = answers.get("text", [])  if isinstance(answers, dict) else []
        ans_scores = answers.get("score", []) if isinstance(answers, dict) else []
        if not ans_texts:
            return None
        if ans_scores and len(ans_scores) == len(ans_texts):
            best   = max(range(len(ans_texts)), key=lambda i: ans_scores[i])
            answer = _clean(ans_texts[best])
        else:
            answer = _clean(ans_texts[0])
        # Trim long ELI5 answers to a readable length
        answer = _trim_answer(answer, MAX_ANSWER_WORDS)
        return context, question, answer

    return None


# Format a single block

def _format_block(context: str, question: str, answer: str,
                  style: str = "answer") -> str:
    context    = _truncate(context, max_words=120)
    completion = (
        "Explanation" if style == "explain" else "Answer"
    )
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"{completion}: {answer}\n"
    )


# Main download + write

def dump_qa_bulk():
    for name, config, split, out_file, n_samples, style in SOURCES:
        print(f"\nProcessing {name} ({'default' if config is None else config}) [{style}]...")
        kwargs = {"streaming": True}
        if config:
            kwargs["name"] = config
        try:
            ds = load_dataset(name, split=split, trust_remote_code=True, **kwargs)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        out_path = QA_DIR / out_file
        count   = 0
        skipped = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for ex in ds:
                result = _extract(ex, name)
                if result is None:
                    skipped += 1
                    continue

                context, question, answer = result

                if not (context and question and answer):
                    skipped += 1
                    continue
                if len(context.split()) < MIN_CONTEXT_WORDS:
                    skipped += 1
                    continue
                if len(answer.split()) < MIN_ANSWER_WORDS:
                    skipped += 1
                    continue

                f.write(_format_block(context, question, answer, style))
                f.write("\n")
                count += 1
                if count >= n_samples:
                    break

        print(f"  Saved {count:,} blocks (skipped {skipped:,}) -> {out_path}")

    print("\nDone.")


# Loader used by finetune.py

def load_all_qa_blocks(source_files=None) -> list[str]:
    if source_files is None:
        source_files = ACTIVE_QA_FILES

    blocks = []
    for file_name in source_files:
        file = QA_DIR / file_name
        if not file.exists():
            continue
        text = file.read_text(encoding="utf-8").strip()
        if text:
            file_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            blocks.extend(file_blocks)

    return blocks


if __name__ == "__main__":
    dump_qa_bulk()