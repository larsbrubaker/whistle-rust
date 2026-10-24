#!/usr/bin/env bash
# Downloads the pinned Whistle weights (REFERENCE "whistle-weights", Apache-2.0) into models/:
# whistle.cact (16.9 MB, what the library loads) and, with --checkpoint, the fp32
# checkpoints/whistle.safetensors (220 MB, the oracle's cross-check). With --engine, also the
# closed-source Needle 3 engine CLI for this host into oracle/.engine/ (REFERENCE
# "needle3-engine"): a black-box transcript oracle only, never disassembled.
# Every file is checked against scripts/weights.sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS=$(awk '$1 == "whistle-weights" { print $2 }' REFERENCE)
ENGINE=$(awk '$1 == "needle3-engine" { print $2 }' REFERENCE)
HF=https://huggingface.co

fetch() { # repo revision path dest
  if [ ! -f "$4" ]; then
    mkdir -p "$(dirname "$4")"
    curl -sSfL "$HF/$1/resolve/$2/$3" -o "$4.part"
    mv "$4.part" "$4"
  fi
  echo "$4"
}

files=(models/whistle.cact)
fetch Cactus-Compute/whistle "$WEIGHTS" whistle.cact models/whistle.cact >/dev/null
for arg in "$@"; do
  case "$arg" in
    --checkpoint)
      files+=(models/whistle.safetensors)
      fetch Cactus-Compute/whistle "$WEIGHTS" checkpoints/whistle.safetensors models/whistle.safetensors >/dev/null ;;
    --engine)
      case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) platform=macos-arm64 runner=needle ;;
        Linux-x86_64) platform=linux-x86_64 runner=needle ;;
        Linux-aarch64) platform=linux-arm64 runner=needle ;;
        *) echo "no published engine for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
      esac
      files+=("oracle/.engine/$platform/$runner")
      fetch Cactus-Compute/needle3 "$ENGINE" "$platform/$runner" "oracle/.engine/$platform/$runner" >/dev/null
      chmod +x "oracle/.engine/$platform/$runner" ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

for f in "${files[@]}"; do
  want=$(awk -v f="$f" '$2 == f { print $1 }' scripts/weights.sha256 2>/dev/null || true)
  have=$(shasum -a 256 "$f" | cut -d' ' -f1)
  if [ -z "$want" ]; then
    echo "$have  $f" >> scripts/weights.sha256
    echo "recorded $f $have"
  elif [ "$want" != "$have" ]; then
    echo "checksum mismatch for $f: want $want, have $have" >&2; exit 1
  else
    echo "ok $f"
  fi
done
