#!/usr/bin/env bash
set -eu
# Use the source shipped in the project's locked MDTraj wheel.
root=$(cd "$(dirname "$0")/.." && pwd)
geometry=$("$PYTHON" -c 'from pathlib import Path; import mdtraj; print(Path(mdtraj.__file__).parent / "geometry")')
out="$root/web/vendor/mdtraj"
em++ -O2 -D__NO_INTRINSICS -I"$geometry/include" -I"$geometry/src/kernels" \
  "$geometry/src/dssp.cpp" "$geometry/src/geometry.cpp" \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=worker,web \
  -sEXPORTED_FUNCTIONS=_dssp,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 \
  -o "$out/dssp.mjs"
# Corresponding library sources, including notices, for this separate module.
tar -czf "$out/source.tar.gz" -C "$geometry" include src/kernels src/dssp.cpp src/geometry.cpp
