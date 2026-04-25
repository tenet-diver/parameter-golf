# train_gpt.py Source Excerpt (PR1797)

Provenance
- Source PR: https://github.com/openai/parameter-golf/pull/1797
- Source path: records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/train_gpt.py
- Head SHA: 04d35edaad74fc88b5ef08a814c94596b616ec1b
- Retrieval command: `mcp__codex_apps__github._fetch_file(.../train_gpt.py, ref=04d35...)`

Runnable train/eval implementation evidence
- Defines `def train_model(h, device, val_data):` with bounded wallclock stopping logic (`max_wallclock_seconds`).
- Defines `def eval_val_ttt_phased(...)` for phased score-before-update TTT evaluation path.
- Defines `def train_and_eval(h, device):` which executes training, quantized eval, and TTT eval path.
- Entry point `def main():` calls `train_and_eval(h, device)` under distributed `torchrun` context.
