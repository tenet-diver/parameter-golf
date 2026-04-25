# Record: PR #1787 base + Smear Gate + LQER Asymmetric + Phased TTT (Imported Excerpt)

Provenance
- Source PR: https://github.com/openai/parameter-golf/pull/1797
- Source path: records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/README.md
- Head SHA: 04d35edaad74fc88b5ef08a814c94596b616ec1b

Runnable command path excerpt
```bash
for SEED in 314 42 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  ...
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

Key reported results
- Mean val_bpb: 1.06157
- artifact bytes max: 15,953,718
- train budget: <= 600s
- eval budget: <= 600s
