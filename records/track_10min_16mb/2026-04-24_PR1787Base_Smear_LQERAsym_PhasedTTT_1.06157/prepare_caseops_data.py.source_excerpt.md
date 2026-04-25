# prepare_caseops_data.py Source Excerpt (PR1797)

Provenance
- Source PR: https://github.com/openai/parameter-golf/pull/1797
- Source path: records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/prepare_caseops_data.py
- Head SHA: 04d35edaad74fc88b5ef08a814c94596b616ec1b

Legality-relevant excerpt
- Imports `surface_piece_original_byte_counts` from `lossless_caps`.
- Defines `BOS_ID = 1` and prepends BOS per document.
- Emits `fineweb_val_bytes_*.bin` sidecar for canonical original-byte BPB accounting.
