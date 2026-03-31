# AI Study Companion with Class-Specific Knowledge

## Quickstart

Use this when you want the shortest path from setup to a working model.

```bash
source ~/venv/bin/activate
cd ~/CS-5822-ChatBot/src
python tokenizer.py --test
nohup python -u train.py --force > ../training.log 2>&1 &
tail -f ../training.log
```

After pretraining completes:

```bash
python finetune_get_data.py
nohup python -u finetune.py --model_file_name Model_best.pth --steps 5000 --batch_size 32 --seq_len 256 --lr 2e-5 --eval_interval 250 --val_ratio 0.05 > ../finetune.log 2>&1 &
tail -f ../finetune.log
```

## Full Training Workflow

### 1 Environment Setup

```bash
source ~/venv/bin/activate
cd ~/CS-5822-ChatBot
```

Recommended:
1. Add Hugging Face token to `.env` as `HF_TOKEN=...` for faster and more reliable dataset access.

### 2 Tokenizer Workflow

Run from `src`:

Dump raw text:

```bash
python tokenizer.py --dump
```

Train tokenizer:

The vocab size is that way since that is what my initial training landed on.
You can change this but reccomended is more the better since hugging face BPE can auto stop when it can't mrege anymore.
```bash
python tokenizer.py --train --vocab_size 44705
```

Validate tokenizer:

```bash
python tokenizer.py --test --text "What is photosynthesis and how does it work?"
```

### 3 Build Token Bins

Convert text from `raw_text` into `.bin` token files in `tokens`:

```bash
python bin_loader.py
```

### 4 Pretrain Base Model

Run in background from `src`:

```bash
nohup python -u train.py --force > ../training.log 2>&1 &
```

Monitor:

```bash
tail -f ../training.log
```

Expected outputs in `model_state`:
1. `Model_best.pth`
2. `Model_final.pth`

### 5) Fine-Tune on QA Data

Build QA dataset (SQuAD format with context):

```bash
python finetune_get_data.py
```

Fine-tune from pretrained best checkpoint:

```bash
nohup python -u finetune.py \
  --model_file_name Model_best.pth \
  --steps 5000 \
  --batch_size 32 \
  --seq_len 256 \
  --lr 2e-5 \
  --eval_interval 250 \
  --val_ratio 0.05 \
  > ../finetune.log 2>&1 &
```

Expected outputs in `model_state` after full pipeline:
1. `Model_best.pth`
2. `Model_final.pth`
3. `Model_finetuned_best.pth`

## Inference and Grounding

### Basic Generation

```bash
python generate.py --model_file_name Model_finetuned_best.pth
```

### Non-Interactive Prompt Mode

```bash
python generate.py --model_file_name Model_finetuned_best.pth --prompt "How hot is the sun?"
```

### Context-Grounded QA (Auto Fallback)

```bash
python generate.py \
  --model_file_name Model_finetuned_best.pth \
  --prompt "How hot is the sun?" \
  --prepend_bos \
  --temperature 0.1 \
  --top_k 10 \
  --max_new_tokens 80 \
  --context "The Sun's photosphere is about 5,500 C, while its core reaches around 15 million C." \
  --instruction "Answer in one sentence and include both photosphere and core temperatures with units." \
  --auto_grounding \
  --grounding_sentences 2 \
  --debug_grounding
```

Grounding modes:
1. `--strict_context_grounding`: always use extractive context grounding.
2. `--auto_grounding`: generate first, then fallback to grounding if reliability checks fail.
3. `--debug_grounding`: print source tag (`model`, `grounded-auto`, `grounded-strict`).