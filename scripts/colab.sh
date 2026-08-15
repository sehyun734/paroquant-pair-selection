#!/usr/bin/env bash

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
session="$(basename "$root")"
gpu="A100"
remote="/content/$session"
stamp="$(TZ=Asia/Seoul date +%y%m%d-%H%M%S)"
tmp="$(mktemp)"

mkdir -p "$root/runs"
colab new -s "$session" --gpu "$gpu"
trap 'rm -f "$tmp"; colab stop -s "$session" >/dev/null 2>&1' EXIT

colab install -s "$session" \
    datasets \
    transformers \
    jaxtyping \
    simple-parsing

tar czf "$tmp" -C "$root" \
    --no-xattrs --no-mac-metadata \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'runs' \
    --exclude '.DS_Store' \
    .

colab upload -s "$session" "$tmp" "$remote.tgz"

colab exec -s "$session" --timeout 300 <<EOF
import subprocess
_ = subprocess.run(
    "mkdir -p $remote && tar xzf $remote.tgz -C $remote && rm $remote.tgz && ls $remote",
    shell=True, check=True,
)
EOF

entry="src/loss.py"
case "${1:-}" in
    *.py) entry="$1"; shift ;;
esac

log="$(basename "$entry" .py)_$stamp.log"
echo "runs/$log" 

colab exec -s "$session" --timeout 86400 <<EOF > "$root/runs/$log"
import subprocess
proc = subprocess.Popen(
    ["bash", "-c", "cd $remote && set -a && [ -f .env ] && . ./.env; set +a; HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 python -u $entry $@ 2>&1"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait():
    raise SystemExit(proc.returncode)
EOF
