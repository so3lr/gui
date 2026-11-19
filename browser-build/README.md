# Rebuilding the browser models

The deployed site serves the checked-in artifacts directly. Each job downloads one pair of compressed files for its model and shape. The force module runs the gradient of the pinned SO3LR potential in float32. The dynamics worker integrates in double precision. The final two atom slots belong to a separate, masked graph. Its padding edges span 200 Å, outside both model cutoffs, to keep distance derivatives finite.

To reproduce the binaries, run `uv sync --locked` in the GUI checkout. Set `PYTHON` to the absolute path of its `.venv/bin/python`. Use a separate Python environment with `iree-base-compiler==3.14.0`, CMake and Ninja on PATH, a native C/C++ toolchain and a fresh build directory. Set `GUI` to the absolute path of the GUI checkout.

```sh
git clone https://github.com/iree-org/iree
git -C iree checkout 5332c68297a68fafd596277d7686f91009ebf1cb
git -C iree submodule update --init third_party/flatcc
git -C iree apply "$GUI/browser-build/inline-notification.patch"
git clone https://github.com/emscripten-core/emsdk
emsdk/emsdk install 4.0.21
emsdk/emsdk activate 4.0.21
export PYTHON="$GUI/.venv/bin/python"
"$GUI/browser-build/build.sh"
```

IREE's synchronous HAL needs an in-memory epoch notification in its single-worker JavaScript proactor. The patch supplies that operation. This build uses a single WebAssembly thread with SIMD. The isolated force model has no periodic electrostatics or cell-stress entry point.

The separate MDTraj module computes the Ala15 DSSP fractions using the same library as the local engine. Its corresponding library sources and license are in `web/vendor/mdtraj/`. To rebuild it separately, put the activated Emscripten toolchain on PATH and run `browser-build/dssp.sh`.

## Browser regression checks

Serve the GUI checkout's parent directory with the web tree mounted at `/gui/` (or set `TEST_URL` to another static subdirectory). Run `node test_browser.cjs` with Playwright available to Node and its Chromium browser installed. Set `BROWSER=webkit` to test with Playwright WebKit. The suite checks force components against float32 ASE reference values, the Ala15 DSSP fractions, and the molecular dynamics, optimization, import, export and cancellation controls.

The reference fixture stores eV and eV/Å for each library geometry as supplied, including its total charge. It uses the locked `So3lrCalculator(model=..., precision="float32", lr_cutoff=100.)`. Tolerances are 0.002 eV for the total energy and 0.001 eV/Å for individual force components.
