import numpy as np
from pathlib import Path
from tokenizer import load
import argparse

def loadBins(file_name=None):
    RAW_TEXT_DIR = Path(__file__).parent.parent / "raw_text"
    TOKENS_DIR = Path(__file__).parent.parent / "tokens"
    TOKENS_DIR.mkdir(exist_ok=True)

    CHUNK_SIZE = 100_000
    BATCH_SIZE = 1000  # number of lines per batch

    tok = load()
    BOS_ID = tok.token_to_id("[BOS]")
    EOS_ID = tok.token_to_id("[EOS]")
    
    txt_files = list(RAW_TEXT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} file(s)")

    if file_name:
        txt_files = [f for f in txt_files if f.name == file_name]

    for txt_file in txt_files:
        out_path = TOKENS_DIR / txt_file.with_suffix(".bin").name
        buffer = []
        total = 0

        with open(out_path, "wb") as out_file:
            with open(txt_file, "r", encoding="utf-8") as f:
                batch = []

                for line in f:
                    line = line.strip()
                    if line:
                        batch.append(line)

                    if len(batch) >= BATCH_SIZE:
                        encodings = tok.encode_batch(batch)

                        for enc in encodings:
                            ids = [BOS_ID] + enc.ids + [EOS_ID]
                            buffer.extend(ids)

                        batch = []

                    if len(buffer) >= CHUNK_SIZE:
                        arr = np.array(buffer, dtype=np.uint16)
                        out_file.write(arr.tobytes())
                        total += len(buffer)
                        buffer = []
                        print(f"{txt_file.name}: {total:,} tokens written...", end="\r")

                # process leftover batch
                if batch:
                    encodings = tok.encode_batch(batch)
                    for enc in encodings:
                        ids = enc.ids
                        ids = [BOS_ID] + ids + [EOS_ID]
                        buffer.extend(ids)

                if buffer:
                    arr = np.array(buffer, dtype=np.uint16)
                    out_file.write(arr.tobytes())
                    total += len(buffer)
                    buffer = []

        print(f"\n{txt_file.name} -> {total:,} tokens saved")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, help="Process only a specific file in raw_text")
    
    args = parser.parse_args()
    
    loadBins(file_name=args.file_name)