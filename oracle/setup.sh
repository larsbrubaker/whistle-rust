#!/usr/bin/env bash
# Creates oracle/.venv (Python 3.12, which JAX supports) for fixture generation only: the Rust
# crates and their tests never call Python. Installs requirements.lock when present so fixtures
# regenerate with the same JAX build; otherwise resolves requirements.txt and writes the lock.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
[ -d .venv ] || "$PY" -m venv .venv
if [ -f requirements.lock ]; then
  .venv/bin/pip install --quiet -r requirements.lock
else
  .venv/bin/pip install --quiet -r requirements.txt
  .venv/bin/pip freeze > requirements.lock
fi
.venv/bin/python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
