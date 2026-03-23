import numpy as np
from pathlib import Path
from tokenizer import load

RAW_TEXT_DIR = Path(__file__).parent.parent / "raw_text"
TOKENS_DIR   = Path(__file__).parent.parent / "tokens"
TOKENS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 100_000  # flush to disk every 100k tokens

tok = load()

txt_files = list(RAW_TEXT_DIR.rglob("*.txt"))
print(f"Found {len(txt_files)} file(s)")

for txt_file in txt_files:
    out_path = TOKENS_DIR / txt_file.with_suffix(".bin").name
    buffer = []
    total = 0

    with open(out_path, "wb") as out_file:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    buffer.extend(tok.encode(line).ids)

                if len(buffer) >= CHUNK_SIZE:
                    arr = np.array(buffer, dtype=np.uint16)
                    arr.tofile(out_file)
                    total += len(buffer)
                    buffer = []
                    print(f"  {txt_file.name}: {total:,} tokens written...", end="\r")

        # flush remainder
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out_file)
            total += len(buffer)

    print(f"\n{txt_file.name} → {total:,} tokens saved")