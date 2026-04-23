# retrieve.py
# Finds the most relevant chunk(s) from a user's notes for a given question
# using TF-IDF scoring. No external dependencies beyond the standard library.

import re
import math
from pathlib import Path
from collections import Counter

USERS_DIR = Path(__file__).parent.parent / "users"

CHUNK_SIZE   = 90    # words per chunk
CHUNK_STRIDE = 40    # overlap between chunks (50% overlap)
MIN_CHUNK_WORDS = 15 # discard very short chunks


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, 2+ chars (removes stopwords implicitly via IDF)."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _preprocess(text: str) -> str:
    """Clean up slide-extracted text before chunking."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip timeline-style bullets (year + word)
        if re.match(r'^[•\-\*]?\s*\d{4}\s+\w', line):
            continue
        # Strip leading bullet characters
        line = re.sub(r'^[•\-\*]\s*', '', line).strip()
        # Skip short lines — slide titles, section headers, lone words
        if len(line.split()) < 5:
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def _is_prose(chunk: str, min_ratio: float = 0.6) -> bool:
    """
    Return True if the chunk looks like readable prose rather than
    diagram notation, math symbols, or slide labels.
    A chunk passes if at least min_ratio of its tokens are real words
    (3+ alphabetic characters).
    """
    words = chunk.split()
    if not words:
        return False
    real = sum(1 for w in words if re.match(r'^[a-zA-Z]{3,}$', w))
    return (real / len(words)) >= min_ratio


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_STRIDE):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        if i + CHUNK_SIZE >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def _build_doc_freqs(tokenized_chunks: list[list[str]]) -> dict[str, int]:
    """Count how many chunks contain each token."""
    df: dict[str, int] = Counter()
    for tokens in tokenized_chunks:
        for t in set(tokens):
            df[t] += 1
    return df


def _tfidf_score(
    query_tokens: list[str],
    chunk_tokens: list[str],
    doc_freqs: dict[str, int],
    n_docs: int,
) -> float:
    chunk_counts = Counter(chunk_tokens)
    score = 0.0
    for token in set(query_tokens):
        tf  = chunk_counts.get(token, 0) / max(len(chunk_tokens), 1)
        idf = math.log((n_docs + 1) / (doc_freqs.get(token, 0) + 1))
        score += tf * idf
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_context(
    username: str,
    question: str,
    top_k: int    = 2,
    max_chars: int = 1000,
) -> str:
    """
    Return the most relevant note chunk(s) for a question.

    Args:
        username:  folder name under USERS_DIR
        question:  the student's question string
        top_k:     number of top chunks to concatenate
        max_chars: hard character limit on returned context

    Returns:
        A single string of context, or "" if no notes found.
    """
    user_dir = USERS_DIR / username
    if not user_dir.exists():
        print(f"[retrieve] No notes folder found for user '{username}' at {user_dir}")
        return ""

    # Collect all chunks from every .txt file in the user's folder
    all_chunks: list[str] = []
    for note_file in sorted(user_dir.glob("*.txt")):
        text = note_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            text = _preprocess(text)
            all_chunks.extend(_chunk_text(text))

    if not all_chunks:
        print(f"[retrieve] No content found in notes for user '{username}'")
        return ""

    # Tokenize everything
    tokenized = [_tokenize(c) for c in all_chunks]
    doc_freqs  = _build_doc_freqs(tokenized)
    query_tokens = _tokenize(question)

    if not query_tokens:
        # No meaningful tokens in the question — return the first chunk as fallback
        return all_chunks[0][:max_chars]

    # Score every chunk
    scored = sorted(
        zip((_tfidf_score(query_tokens, toks, doc_freqs, len(all_chunks))
             for toks in tokenized),
            all_chunks),
        reverse=True,
    )

    # Take top_k prose chunks, join with separator, trim to max_chars
    prose_chunks = [chunk for _, chunk in scored if _is_prose(chunk)]
    # Fall back to top scored if nothing passes the prose filter
    if not prose_chunks:
        prose_chunks = [chunk for _, chunk in scored]
    top_chunks = prose_chunks[:top_k]
    context    = " [...] ".join(top_chunks)
    return context[:max_chars].strip()


def format_prompt(username: str, question: str) -> str:
    """
    Build the [NOTE_QA] prompt ready to pass to generate().
    Falls back to a bare [NOTE_QA] block if no context is found.
    """
    context = retrieve_context(username, question)
    if context:
        return (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    else:
        return (
            f"Context: No relevant notes found.\n"
            f"Question: {question}\n"
            f"Answer:"
        )


# ---------------------------------------------------------------------------
# CLI — quick test without the full model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test retrieval without running the model.")
    parser.add_argument("--username", required=True, help="Username (folder under users/)")
    parser.add_argument("--question", required=True, help="Question to retrieve context for")
    parser.add_argument("--top_k",    type=int, default=2)
    parser.add_argument("--max_chars",type=int, default=400)
    args = parser.parse_args()

    print("\n--- Retrieved Context ---")
    ctx = retrieve_context(args.username, args.question, args.top_k, args.max_chars)
    print(ctx if ctx else "(none)")

    print("\n--- Full Prompt ---")
    print(format_prompt(args.username, args.question))