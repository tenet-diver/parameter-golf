#!/usr/bin/env bash
set -euo pipefail

PARAMETER_GOLF_REPO_URL="${PARAMETER_GOLF_REPO_URL:-https://github.com/tenet-diver/parameter-golf.git}"
PARAMETER_GOLF_REF="${PARAMETER_GOLF_REF:-develop}"
PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-/workspace/parameter-golf}"
FASTEST_REPO_URL="${FASTEST_REPO_URL:-https://github.com/power-of-positive/fastest.git}"
FASTEST_REF="${FASTEST_REF:-develop}"
FASTEST_DIR="${FASTEST_DIR:-/workspace/fastest}"

CLONE_FASTEST="${CLONE_FASTEST:-false}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-true}"
PG_TRAIN_SHARDS="${PG_TRAIN_SHARDS:-80}"
INSTALL_TORCH="${INSTALL_TORCH:-true}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_CODEX="${INSTALL_CODEX:-false}"
RUN_GH_AUTH="${RUN_GH_AUTH:-false}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
else
  SUDO=(sudo)
fi

log() {
  printf '\n==> %s\n' "$*"
}

bool_true() {
  case "${1:-}" in
    1 | true | TRUE | yes | YES | y | Y) return 0 ;;
    *) return 1 ;;
  esac
}

install_gh_cli() {
  if command -v gh >/dev/null 2>&1; then
    return
  fi
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | "${SUDO[@]}" dd of=/etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
  "${SUDO[@]}" chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y gh
}

configure_github_auth() {
  local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [ -n "$token" ]; then
    if ! gh auth status >/dev/null 2>&1; then
      printf '%s' "$token" | gh auth login --with-token || true
    fi
    gh auth setup-git || true
    return
  fi

  if bool_true "$RUN_GH_AUTH"; then
    gh auth login --web --git-protocol https
    gh auth setup-git || true
  else
    cat <<'EOF'
GitHub auth was not forced. Public HTTPS clones will work as-is.

For a private fork, rerun with one of:
  GITHUB_TOKEN=ghp_... bash scripts/setup_runpod_parameter_golf_deadline.sh
  RUN_GH_AUTH=true bash scripts/setup_runpod_parameter_golf_deadline.sh
EOF
  fi
}

clone_or_update() {
  local repo_url="$1"
  local ref="$2"
  local dest="$3"
  local label="$4"

  mkdir -p "$(dirname "$dest")"
  if [ -d "$dest/.git" ]; then
    log "Updating $label in $dest"
    git -C "$dest" fetch --all --tags --prune || true
    if git -C "$dest" rev-parse --verify "origin/$ref" >/dev/null 2>&1; then
      git -C "$dest" checkout "$ref"
      git -C "$dest" pull --ff-only origin "$ref" || true
    else
      git -C "$dest" pull --ff-only || true
    fi
    return
  fi

  log "Cloning $label into $dest"
  if git ls-remote --exit-code --heads "$repo_url" "$ref" >/dev/null 2>&1; then
    git clone --branch "$ref" "$repo_url" "$dest"
  else
    echo "Branch '$ref' was not found for $repo_url; cloning the remote default branch."
    git clone "$repo_url" "$dest"
  fi
}

log "Installing system tools"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  git git-lfs curl ca-certificates gnupg tmux htop nvtop ripgrep \
  build-essential python3-venv python3-pip
install_gh_cli
git lfs install || true

log "Configuring GitHub auth if credentials were provided"
configure_github_auth

if bool_true "$INSTALL_CODEX"; then
  log "Installing Node.js 22 and Codex CLI"
  if ! command -v npm >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | "${SUDO[@]}" bash -
    "${SUDO[@]}" apt-get install -y nodejs
  fi
  npm i -g @openai/codex
  codex --version || true
fi

log "Setting cache locations under /workspace"
mkdir -p /workspace/.cache/huggingface/datasets
if ! grep -q 'HF_HOME=/workspace/.cache/huggingface' /root/.bashrc 2>/dev/null; then
  cat >> /root/.bashrc <<'EOF'

export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
EOF
fi
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

clone_or_update "$PARAMETER_GOLF_REPO_URL" "$PARAMETER_GOLF_REF" "$PARAMETER_GOLF_DIR" "Parameter Golf"
cd "$PARAMETER_GOLF_DIR"

log "Installing Parameter Golf Python dependencies"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
if bool_true "$INSTALL_TORCH"; then
  python3 -m pip install --upgrade --force-reinstall \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"
fi

log "Verifying CUDA, H100 visibility, and grouped-query attention"
nvidia-smi -L
python3 - <<'PY'
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
q = torch.randn(1, 4, 8, 16, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.bfloat16)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0), "gqa_ok", tuple(y.shape))
PY

if bool_true "$DOWNLOAD_DATA"; then
  log "Downloading Parameter Golf dataset, train shards: $PG_TRAIN_SHARDS"
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$PG_TRAIN_SHARDS"
fi

if bool_true "$CLONE_FASTEST"; then
  clone_or_update "$FASTEST_REPO_URL" "$FASTEST_REF" "$FASTEST_DIR" "Fastest"
  if [ -f "$FASTEST_DIR/package.json" ]; then
    log "Installing and building Fastest"
    cd "$FASTEST_DIR"
    npm ci
    npm run build:orchestrator --silent || true
    npm run build:scripts --silent || true
    cd "$PARAMETER_GOLF_DIR"
  fi
fi

if [ ! -f "$PARAMETER_GOLF_DIR/scripts/run_deadline_experiment_batch.sh" ]; then
  cat >&2 <<EOF
Missing $PARAMETER_GOLF_DIR/scripts/run_deadline_experiment_batch.sh.

The cloned ref does not contain the deadline batch runner. Push this branch first,
or rerun setup with PARAMETER_GOLF_REF set to the branch that contains it.
EOF
  exit 1
fi
chmod +x "$PARAMETER_GOLF_DIR/scripts/run_deadline_experiment_batch.sh"

log "GPU experiment queue inventory"
python3 fastest/scripts/run_gpu_experiment_queue.py --print-inventory --dry-run --max-experiments 12 || true

cat > /workspace/pg_deadline_env.sh <<EOF
export PARAMETER_GOLF_DIR="$PARAMETER_GOLF_DIR"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
EOF

cat <<EOF

==> Setup complete

Parameter Golf: $PARAMETER_GOLF_DIR
Environment:    /workspace/pg_deadline_env.sh

Recommended first pass on a 1xH100 pod:
  cd "$PARAMETER_GOLF_DIR"
  source /workspace/pg_deadline_env.sh
  tmux new -s pg1 'PG_GPUS=1 PG_MAX_EXPERIMENTS=12 scripts/run_deadline_experiment_batch.sh'

After selecting winners, run the 8xH100 phase with explicit IDs:
  cd "$PARAMETER_GOLF_DIR"
  source /workspace/pg_deadline_env.sh
  tmux new -s pg8 'PG_GPUS=8 PG_ONLY=id1,id2 PG_PACKAGE_SUBMISSION=1 PG_SUBMISSION_AUTHOR="Your Name" PG_SUBMISSION_GITHUB_ID="your-handle" scripts/run_deadline_experiment_batch.sh'

Useful overrides:
  PG_TRAIN_SHARDS=1       # faster setup smoke only
  PG_DRY_RUN=1            # inspect selected experiments
  PG_LANES=h100-1x-clean-ablation,h100-1x-parameter-tuning
  PG_STATUS=ready         # skip queued experiments
  PG_PACKAGE_SUBMISSION=1 # create a PR-ready records/track_10min_16mb folder from the best completed run
EOF
