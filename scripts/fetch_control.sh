#!/usr/bin/env bash
# Pulls the 9.46 GB control checkpoint from HF into the repo.
#
# Safe to run WHILE the Phase A driver is evaluating the 6.02 GB checkpoint:
# this is network+disk bound, that is GPU bound. The driver checks for this
# directory only after it finishes the 6.02 GB tasks, so a download that lands
# in time is picked up automatically in the same overnight run.
set -uo pipefail

REPO="$HOME/zaya1-godspeed"
DEST="$REPO/zaya1-8b-nvfp4-w4a4"
REPO_ID="Ttimms/zaya1-8b-nvfp4-w4a4"

cd "$REPO" || exit 1
source "$HOME/vllm-env/bin/activate" || exit 1

echo "[$(date +%H:%M:%S)] verifying auth before a ~9.5 GB transfer"
python3 - <<'PY'
from huggingface_hub import whoami
try:
    me = whoami()
    print(f"  auth OK: {me.get('name')}")
except Exception as e:
    print(f"  auth FAILED: {type(e).__name__}")
    print("  run:  hf auth login   (in your own terminal - never paste a token into a Claude session)")
    raise SystemExit(1)
PY
[ $? -ne 0 ] && exit 1

echo "[$(date +%H:%M:%S)] downloading -> $DEST"
hf download "$REPO_ID" --local-dir "$DEST"
rc=$?
echo "[$(date +%H:%M:%S)] download rc=$rc"

if [ $rc -eq 0 ]; then
  bytes=$(find "$DEST" -name '*.safetensors' -printf '%s\n' 2>/dev/null | paste -sd+ | bc)
  echo "  safetensors total: $(echo "scale=2; ${bytes:-0}/1073741824" | bc) GiB"
  ls -1 "$DEST" | sed 's/^/  /'
fi
exit $rc
