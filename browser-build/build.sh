#!/usr/bin/env bash
# Execute this from a build directory with iree/ and emsdk/ as described in README.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out="$src/../web/browser-runtime"
work=$PWD
export PATH="$work/emsdk/upstream/emscripten:$PATH"
export EM_CONFIG="$work/emsdk/.emscripten"
runtime_flags=(
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_COMPILER=OFF
  -DIREE_BUILD_BENCHMARKS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF
  -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF
)
cmake -S iree -B host "${runtime_flags[@]}" -DIREE_HAL_EXECUTABLE_LOADER_STATIC_LIBRARY=ON
cmake --build host --target iree-flatcc-cli iree-c-embed-data
mkdir -p "$out"
for model in s m l; do
  for shape in 16:240 32:992 80:2400 258:8192; do
    atoms=${shape%:*}
    neighbors=${shape#*:}
    "$PYTHON" "$src/export.py" "$atoms" "$neighbors" "$model" force.mlir
    iree-compile force.mlir --iree-hal-target-backends=llvm-cpu \
      --iree-llvmcpu-target-triple=wasm32-unknown-emscripten \
      --iree-llvmcpu-target-cpu-features=+simd128 --iree-llvmcpu-link-embedded=false \
      --iree-llvmcpu-link-static --iree-llvmcpu-static-library-output-path=force.o -o force.vmfb
    cmake -S "$src" -B wasm "${runtime_flags[@]}" -DIREE_HOST_BIN_DIR="$work/host/tools" \
      -DCMAKE_TOOLCHAIN_FILE="$work/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake" \
      -DIREE_ENABLE_THREADING=OFF -DCMAKE_C_FLAGS="-matomics -mbulk-memory" -DCMAKE_CXX_FLAGS="-matomics -mbulk-memory"
    cmake --build wasm --target force
    emsdk/upstream/bin/llvm-objcopy --strip-debug wasm/force.wasm force-stripped.wasm
    gzip -cn9 force.vmfb > "$out/$model-$atoms.vmfb.gz"
    gzip -cn9 force-stripped.wasm > "$out/$model-$atoms.wasm.gz"
  done
done
cp wasm/force.mjs "$out/force-runtime.mjs"
cp iree/runtime/src/iree/async/platform/js/proactor_imports.mjs "$out/proactor.mjs"
"$src/dssp.sh"
