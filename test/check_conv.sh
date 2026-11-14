#!/usr/bin/env bash
# Gate T2: compile a kernel version against the HLS stubs and run it on the
# test vectors, compared bit for bit with the golden model.
#
#   test/check_conv.sh <kernel.cpp> [case ...|all]
#
#   test/check_conv.sh kernel/src/pl_v0.cpp              # every case except real_frame
#   test/check_conv.sh kernel/src/pl_v0.cpp all          # including real_frame (slow on V0)
#   test/check_conv.sh kernel/src/pl_v0.cpp impulse_center rounding
#
# Vectors come from `make vectors` (python3 -m model.gen_vectors).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VEC="$ROOT/test/vectors"
OUT="$ROOT/build/t2"
KERNEL="${1:-}"
shift || true

if [[ -z "$KERNEL" || ! -f "$KERNEL" ]]; then
    echo "usage: test/check_conv.sh <kernel.cpp> [case ...|all]" >&2
    exit 2
fi
if [[ ! -f "$VEC/manifest.csv" ]]; then
    echo "no test vectors in $VEC; run 'make vectors' first" >&2
    exit 2
fi

CXX="${CXX:-c++}"
FLAGS=(-std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
       -I"$ROOT/test/include" -I"$ROOT/golden" -I"$ROOT/kernel/src")
mkdir -p "$OUT"

echo "=== compile $(basename "$KERNEL")"
"$CXX" "${FLAGS[@]}" "$KERNEL" "$ROOT/test/tb_conv.cpp" "$ROOT/golden/espcn_golden.cpp" -o "$OUT/tb_conv"

# Select cases from the manifest: name:H:W
specs=()
while IFS=, read -r name h w _rest; do
    [[ "$name" == "case" ]] && continue
    if [[ $# -eq 0 ]]; then
        [[ "$name" == "real_frame" ]] && continue
    elif [[ "$1" != "all" ]]; then
        match=0
        for want in "$@"; do [[ "$want" == "$name" ]] && match=1; done
        [[ $match -eq 0 ]] && continue
    fi
    specs+=("$name:$h:$w")
done < "$VEC/manifest.csv"

if [[ ${#specs[@]} -eq 0 ]]; then
    echo "no matching cases in $VEC/manifest.csv" >&2
    exit 2
fi

echo "=== run ${#specs[@]} case(s)"
if "$OUT/tb_conv" "$VEC" "${specs[@]}"; then
    echo "ALL LOCAL CHECKS PASSED"
else
    echo "LOCAL CHECK FAILED - do not upload"
    exit 1
fi
