import numpy as np
from pathlib import Path
from tokenizer import load

RAW_TEXT_DIR = Path(__file__).parent.parent / "raw_text"
TOKENS_DIR   = Path(__file__).parent.parent / "tokens"
TOKENS_DIR.mkdir(exist_ok=True)

tok = load()

txt_files = list(RAW_TEXT_DIR.rglob("*.txt"))
print(f"Found {len(txt_files)} file(s)")

for txt_file in txt_files:
    all_ids = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_ids.extend(tok.encode(line).ids)

    arr = np.array(all_ids, dtype=np.uint16)
    out_path = TOKENS_DIR / txt_file.with_suffix(".bin").name
    arr.tofile(out_path)
    print(f"{txt_file.name} → {len(arr):,} tokens saved")