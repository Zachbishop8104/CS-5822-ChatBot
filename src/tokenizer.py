"""
tokenizer.py. Train and load a BPE tokenizer using HuggingFace tokenizers.

Usage:
    python tokenizer.py --dump  --samples #        # step 1: pull raw text
    python tokenizer.py --train --vocab_size #     # step 2: train tokenizer
    python tokenizer.py --test                     # verify it works
"""

import argparse
import logging
import os
from pathlib import Path
from urllib.request import urlretrieve
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.decoders import BPEDecoder
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

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
def dump_texts():
    """Save data from HuggingFace datasets as raw text files in raw_text/."""
    output_dir = RAW_TEXT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sources: (dataset_name, config, split, text_field, samples)
    sources = [
        ("wikimedia/wikipedia", "20231101.en", "train", "text", 3_000_000),
        ("HuggingFaceTB/cosmopedia", "stanford", "train", "text", 500_000),
        ("squad_v2", None, "train", "context", 100_000),
        ("zelalt/scientific-papers", None, "train", "full_text", 200_000),
        ("sci-datasets/sci-papers", None, "train", "text", 500_000),
        ("openwebtext", None, "train", "text", 1_000_000),
        ("cc_news", None, "train", "text", 500_000),
    ]

    for name, config, split, field, n_samples in sources:
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

        log.info(f"Saved {count} samples -> {out_path}")

# Train Byte Pair Encoding tokenizer
# using huggingface tokenizers library
# This takes an hour or two
def train(vocab_size: int = 8000, log_every: int = 1_000_000, include_qa: bool = False):
    """
    Train a BPE tokenizer on the dumped raw text files.

    Args:
        vocab_size (int): target vocabulary size
        log_every (int): log a message every `log_every` lines processed
    """
    txt_files = [
        p for p in RAW_TEXT_DIR.rglob("*.txt")
        if include_qa or "qa" not in p.relative_to(RAW_TEXT_DIR).parts
    ]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_TEXT_DIR}. Run --dump first.")

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    qa_mode = "including QA files" if include_qa else "excluding raw_text/qa"
    log.info(f"Training on {len(txt_files)} file(s) with vocab_size={vocab_size} ({qa_mode})")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        end_of_word_suffix="</w>",
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    def file_generator(files):
        for f_idx, f in enumerate(files, start=1):
            with open(f, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, start=1):
                    if i % log_every == 0:
                        log.info(f"File {f_idx}/{len(files)}: processed {i:,} lines from {f.name}")
                    yield line
    lines_iter = file_generator(txt_files)

    tokenizer.train_from_iterator(lines_iter, trainer=trainer)
    # Match decode behavior to BPE end-of-word markers to restore spaces.
    tokenizer.decoder = BPEDecoder(suffix="</w>")
    tokenizer.save(str(TOKENIZER_PATH))
    log.info(f"Tokenizer saved: {TOKENIZER_PATH}  (vocab_size={tokenizer.get_vocab_size()})")


# Load helper
def load() -> Tokenizer:
    """Load the tokenizer from disk."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    log.info(f"Loaded tokenizer from: {TOKENIZER_PATH} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


# got sick and tired of retaining this thing so added a download option to pull the pretrained tokenizer.json from Hugging Face
def download_pretrained(model_id: str = "openai-community/gpt2"):
    """Download tokenizer.json from Hugging Face and ensure project special tokens exist."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{model_id}/resolve/main/tokenizer.json"
    log.info(f"Downloading tokenizer from: {url}")
    urlretrieve(url, str(TOKENIZER_PATH))

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    missing = [tok for tok in SPECIAL_TOKENS if tokenizer.token_to_id(tok) is None]
    if missing:
        tokenizer.add_special_tokens(missing)
        tokenizer.save(str(TOKENIZER_PATH))
        log.info(f"Added missing special tokens: {missing}")

    log.info(
        f"Tokenizer ready at: {TOKENIZER_PATH} "
        f"(vocab_size={tokenizer.get_vocab_size()}, model_id={model_id})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true", help="Dump raw text from HuggingFace")
    parser.add_argument("--train", action="store_true", help="Train the tokenizer")
    parser.add_argument("--download", action="store_true", help="Download a pretrained tokenizer.json")
    parser.add_argument(
        "--model_id",
        default="openai-community/gpt2",
        help="Hugging Face model id used with --download (must provide tokenizer.json).",
    )
    parser.add_argument("--test", action="store_true", help="Test the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=40_000, help="Vocabulary size for training the tokenizer")
    parser.add_argument(
        "--include_qa",
        action="store_true",
        help="Include raw_text/qa/*.txt files during tokenizer training (default: excluded).",
    )
    parser.add_argument("--text", default="What is photosynthesis and how does it work?")
    args = parser.parse_args()

    if args.dump:
        dump_texts()

    if args.train:
        train(args.vocab_size, include_qa=args.include_qa)

    if args.download:
        download_pretrained(args.model_id)

    if args.test:
        tokenizer = load()
        ids = tokenizer.encode(args.text).ids
        decoded = tokenizer.decode(ids)
        print(f"\nInput: {args.text}")
        print(f"Tokens: {ids}")
        print(f"Decoded: {decoded}")
        print(f"Vocab size: {tokenizer.get_vocab_size()}")