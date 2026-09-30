#!/bin/bash
# Clone RLinf at the commit this work was built on and add the DGPO changes.
#   bash scripts/setup_rlinf.sh            # copy rlinf_overlay/ into RLinf/
#   bash scripts/setup_rlinf.sh --patch    # or: git apply patches/rlinf_dgpo.patch
set -euo pipefail
cd "$(dirname "$0")/.."
RLINF_COMMIT=09a74045c30b934e7f0811c8a550a1b126332da1

if [ ! -d RLinf/.git ]; then
    git clone https://github.com/RLinf/RLinf.git RLinf
fi
git -C RLinf checkout -q $RLINF_COMMIT
git -C RLinf checkout -q -b dgpo 2>/dev/null || git -C RLinf checkout -q dgpo

if [ "${1:-}" = "--patch" ]; then
    git -C RLinf apply --whitespace=nowarn ../patches/rlinf_dgpo.patch
else
    cp -r rlinf_overlay/. RLinf/
fi
echo "RLinf @ ${RLINF_COMMIT:0:8} + DGPO:"
git -C RLinf status --short
