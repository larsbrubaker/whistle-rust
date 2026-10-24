#!/usr/bin/env bash
# Clones cactus-compute/needle (Apache-2.0) at the commit pinned in REFERENCE into
# needle-reference/ (git-ignored). Its needle/model/{whistle,architecture,quantize,export}.py are
# the spec this port follows. Reading material and the oracle's model code; the Rust build never
# uses it. Never fetch or read github.com/cactus-compute/cactus (non-commercial license).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(awk '$1 == "needle" { print $2 }' REFERENCE)
if [ ! -d needle-reference/.git ]; then
  git clone --quiet --filter=blob:none https://github.com/cactus-compute/needle.git needle-reference
fi
git -C needle-reference fetch --quiet origin "$COMMIT"
git -C needle-reference checkout --quiet --detach "$COMMIT"
echo "needle-reference at $COMMIT"
