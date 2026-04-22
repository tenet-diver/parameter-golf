# Non-Record Submission: SP1024 Parallel Residual + Recurrence CPU Control

This folder documents a small, reproducible negative-result control around two motifs that keep showing up in stronger `parameter-golf` runs:

1. parallel residual execution
2. recurrent / virtual-depth layer ordering

Important scope note:

- This is a **non-record research submission**
- This run uses **CPU only**
- This run uses **1 train shard**
- This run evaluates on the **first 1,048,576 validation tokens only** via `VAL_TOKEN_LIMIT=1048576`
- This is **not** a leaderboard claim on the full validation split

## Why submit this

The value here is not raw score. The value is a cheap, honest control showing what happens when these two motifs are enabled in a tiny SP1024 trainer without hiding behind extra systems tricks or larger compute. It is intended as a documented negative result and a reproducible local ablation harness.

## Configuration

- Track: `non-record-16mb`
- Tokenizer/data: published `sp1024` challenge assets
- Hardware: CPU
- Model: 2 layers, width 128, 4 query heads, 2 KV heads
- MLP multiplier: 2
- Sequence length: 128
- Train batch tokens: 8192
- Iterations: 2
- Parallel residual: enabled
- Encoder layer order: `[0]`
- Decoder layer order: `[1, 0, 1]`
- Validation prefix: `1,048,576` tokens

This creates a tiny virtual-depth experiment where one encoder block is followed by a recurrent decoder schedule. It is intentionally small and cheap.

## Results

Exact post-quant replay on the emitted `final_model.int8.ptz`:

- `val_loss`: `6.92837095`
- `val_bpb`: `4.15111081`
- `bytes_model_int8_zlib`: `413652`
- `bytes_code`: `51575`
- `bytes_total`: `465227`

Timing:

- training/serialization pass from `train.log`: `6.173s`
- exact post-quant replay from `eval.log`: `53.280s`
- combined local wallclock used in `submission.json`: `59.453s`

## Takeaway

This control is a clear negative result. Simply turning on parallel residual structure plus a recurrent layer order in a very small CPU-bound SP1024 model is nowhere near competitive by itself. These motifs likely need much stronger training throughput, larger budgets, and better surrounding optimization choices before they become useful.

## Reproduction

Run from the repository root with the published SP1024 assets already downloaded:

```bash
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
RUN_ID=cpu_parres_recur_subset_complete \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=2 \
MODEL_DIM=128 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
TRAIN_SEQ_LEN=128 \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=524288 \
VAL_TOKEN_LIMIT=1048576 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
WARMUP_STEPS=0 \
ITERATIONS=2 \
MAX_WALLCLOCK_SECONDS=0 \
PARALLEL_RESIDUAL=1 \
ENCODER_LAYER_ORDER=0 \
DECODER_LAYER_ORDER=1,0,1 \
SKIP_PRE_QUANT_FINAL_EVAL=1 \
python train_gpt.py
```

`train.log` captures the automated training/serialization pass. `eval.log` captures the exact roundtrip evaluation on the emitted compressed artifact.

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train.log`
- `eval.log`
- `requirements.txt`
