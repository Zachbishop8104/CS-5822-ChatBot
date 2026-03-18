"""
tokenizer.py. Train and load a BPE tokenizer using HuggingFace tokenizers.

Usage:
    python tokenizer.py --dump  --samples #        # step 1: pull raw text
    python tokenizer.py --train --vocab_size #     # step 2: train tokenizer
    python tokenizer.py --test                     # verify it works
"""

import argparse
import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

RAW_TEXT_DIR  = Path(__file__).parent.parent / "raw_text"
TOKENIZER_DIR = Path(__file__).parent.parent / "tokenizer"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"


def _coerce_to_text(value) -> str:
    """Normalize dataset field values into plain text for dumping."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return " ".join(parts).strip()
    if isinstance(value, dict):
        parts = [str(v).strip() for v in value.values() if v is not None and str(v).strip()]
        return " ".join(parts).strip()
    return str(value).strip()

# Dump raw text to disk
def dump_texts(n_samples: int = 10_000):
    """Save data from HuggingFace datasets as raw text files in raw_text/."""
    output_dir = RAW_TEXT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        # (dataset_name, config, split, text_field)
        ("wikipedia", "20220301.en", "train", "text"),
        ("squad_v2", None, "train", "context"), # Stanford Question Answering Dataset
        ("scientific_papers",  "arxiv",       "train", "abstract"),
        ("openbookqa",         "main",        "train", "question_stem"),
        ("blended_skill_talk", None,          "train", "previous_utterance"),
    ]

    for name, config, split, field in sources:
        log.info(f"Dumping {n_samples} samples from {name}...")
        kwargs = {"streaming": True}
        if config:
            kwargs["name"] = config

        ds = load_dataset(name, split=split, **kwargs)
        out_path = output_dir / f"{name.replace('/', '_')}.txt"

        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in ds:
                text = _coerce_to_text(ex.get(field, ""))
                if text:
                    f.write(text + "\n\n")
                    count += 1
                if count >= n_samples:
                    break

        log.info(f"Saved {count} samples → {out_path}")

# Train Byte Pair Encoding tokenizer
# using huggingface tokenizers library
def train(vocab_size: int = 8000):
    txt_files = list(RAW_TEXT_DIR.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_TEXT_DIR}. Run --dump first.")

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Training on {len(txt_files)} file(s) with vocab_size={vocab_size}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train([str(f) for f in txt_files], trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    log.info(f"Tokenizer saved: {TOKENIZER_PATH}  (vocab_size={tokenizer.get_vocab_size()})")


# Load helper
def load() -> Tokenizer:
    """Load the tokenizer from disk."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    log.info(f"Loaded tokenizer from: {TOKENIZER_PATH} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true", help="Dump raw text from HuggingFace")
    parser.add_argument("--train", action="store_true", help="Train the tokenizer")
    parser.add_argument("--test", action="store_true", help="Test the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--text", default="What is photosynthesis and how does it work?")
    args = parser.parse_args()

    if args.dump:
        dump_texts(args.n_samples)

    if args.train:
        train(args.vocab_size)

    if args.test:
        tokenizer = load()
        ids = tokenizer.encode(args.text).ids
        decoded = tokenizer.decode(ids)
        print(f"\nInput: {args.text}")
        print(f"Tokens: {ids}")
        print(f"Decoded: {decoded}")
        print(f"Vocab size: {tokenizer.get_vocab_size()}")