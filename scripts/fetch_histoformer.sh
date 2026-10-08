#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
destination=${1:-"$root/third_party/Histoformer"}
commit=1f045f06c03551c31504d8042dbe6ff9b9569108
arch_relative=basicsr/models/archs/histoformer_arch.py
arch_sha256=2480609a85c1d1c02144743f992bf4f8c9e979ac5cb7126f1c2e02be956670b0

if [[ -e "$destination" ]]; then
  echo "Refusing existing destination: $destination" >&2
  exit 2
fi

git clone https://github.com/sunshangquan/Histoformer.git "$destination"
git -C "$destination" checkout --detach "$commit"

actual_commit=$(git -C "$destination" rev-parse HEAD)
if [[ "$actual_commit" != "$commit" ]]; then
  echo "Histoformer commit drift: $actual_commit" >&2
  exit 2
fi

arch_path="$destination/$arch_relative"
actual_sha256=$(python3 - "$arch_path" <<'PY'
from pathlib import Path
import hashlib
import sys

path = Path(sys.argv[1]).resolve(strict=True)
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)
if [[ "$actual_sha256" != "$arch_sha256" ]]; then
  echo "Histoformer architecture SHA drift: $actual_sha256" >&2
  exit 2
fi

if git -C "$destination" ls-tree -r --name-only HEAD \
  | grep -Eiq '(^|/)(license|copying|notice)(\.|$)'; then
  echo "Upstream license/notice file detected; review it before release." >&2
else
  cat >&2 <<'EOF'
WARNING: the pinned upstream tree has no repository-level LICENSE/COPYING/
NOTICE file. This project does not redistribute that source or assign it an
Apache license. Obtain permission or review upstream terms before public use.
EOF
fi

echo "Pinned Histoformer ready: $destination"
echo "commit=$actual_commit"
echo "architecture_sha256=$actual_sha256"
