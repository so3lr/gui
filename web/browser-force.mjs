import { createImports } from './browser-runtime/proactor.mjs';

async function decompress(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error('Could not load the browser model (' + response.status + ').');
  // Some static hosts deliver .gz as a file; others set Content-Encoding
  // and Fetch has already decompressed the response.
  const bytes = await response.arrayBuffer();
  const header = new Uint8Array(bytes, 0, Math.min(2, bytes.byteLength));
  if (header[0] !== 0x1f || header[1] !== 0x8b) return bytes;
  const decoded = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(decoded).arrayBuffer();
}

async function loadKernel(model, n, edges) {
  const prefix = new URL('./browser-runtime/' + model + '-' + n, import.meta.url).href;
  const [glue, bytes, weights] = await Promise.all([
    import('./browser-runtime/force-runtime.mjs'),
    decompress(prefix + '.wasm.gz'),
    decompress(prefix + '.vmfb.gz'),
  ]);
  const compiled = await WebAssembly.compile(bytes);
  const context = { inline: true };
  const runtime = await glue.default({
    instantiateWasm(imports, receive) {
      imports.iree_proactor = createImports(context);
      const instance = new WebAssembly.Instance(compiled, imports);
      context.memory = instance.exports.memory;
      receive(instance, compiled);
      return instance.exports;
    },
  });
  const blob = runtime._malloc(weights.byteLength);
  runtime.HEAPU8.set(new Uint8Array(weights), blob);
  if (runtime._setup(blob, weights.byteLength))
    throw Error('Could not initialize SO3LR on this device.');
  const long = (n * (n - 1)) / 2;
  const [r, z, q, sr, lr, e, g] = [3 * n, n, 1, 2 * edges, 2 * long, 1, 3 * n].map((size) =>
    runtime._malloc(4 * size),
  );
  return function evaluate(x, species, charge, shortI, shortJ, longI, longJ) {
    const f = runtime.HEAPF32,
      i = runtime.HEAP32;
    f.fill(0, r / 4, r / 4 + 3 * n);
    f.set(x, r / 4);
    // Place the two padding endpoints beyond both interaction cutoffs.
    f[r / 4 + 3 * (n - 1)] = 200;
    i.fill(0, z / 4, z / 4 + n);
    i.set(species, z / 4);
    f[q / 4] = charge;
    for (const [ptr, capacity, first, second] of [
      [sr, edges, shortI, shortJ],
      [lr, long, longI, longJ],
    ]) {
      i.fill(n - 2, ptr / 4, ptr / 4 + capacity);
      i.fill(n - 1, ptr / 4 + capacity, ptr / 4 + 2 * capacity);
      i.set(first, ptr / 4);
      i.set(second, ptr / 4 + capacity);
    }
    if (runtime._evaluate(n, edges, long, r, z, q, sr, lr, e, g))
      throw Error('SO3LR force evaluation failed.');
    const energy = runtime.HEAPF32[e / 4];
    const forces = Float64Array.from(runtime.HEAPF32.subarray(g / 4, g / 4 + x.length), (v) => -v);
    if (!Number.isFinite(energy) || !forces.every(Number.isFinite))
      throw Error('Non-finite model output. Check the geometry and time step.');
    return { energy, forces };
  };
}

export function createForceEvaluator(model, species, charge) {
  const key = { 'so3lr-tauri-s': 's', 'so3lr-tauri-m': 'm', 'so3lr-tauri-l': 'l' }[model];
  if (!key) throw Error('Unknown SO3LR model.');
  const cutoff = key === 'l' ? 5 : 4.5;
  // Select the smallest suitable workload; denser geometries can promote
  // to a larger kernel without dropping neighbors. L has a longer cutoff.
  const buckets = [
    [8, 32],
    [16, 240],
    [18, 224],
    [34, 640],
    [32, 992],
    [66, 1536],
    [80, 2400],
    [166, key === 'l' ? 5632 : 4352],
    [258, 8192],
  ];
  let capacity = 0,
    kernel;
  return async (x) => {
    const si = [],
      sj = [],
      li = [],
      lj = [];
    for (let i = 0; i < species.length; i++)
      for (let j = i + 1; j < species.length; j++) {
        const distance = Math.hypot(
          x[3 * i] - x[3 * j],
          x[3 * i + 1] - x[3 * j + 1],
          x[3 * i + 2] - x[3 * j + 2],
        );
        if (distance < 0.0001) throw Error('Two atoms occupy the same position.');
        if (distance < 100) {
          li.push(i);
          lj.push(j);
        }
        if (distance < cutoff) {
          si.push(i, j);
          sj.push(j, i);
        }
      }
    if (!kernel || si.length > capacity) {
      const bucket = buckets.find(
        ([atoms, edges]) => species.length + 2 <= atoms && si.length <= edges,
      );
      if (!bucket)
        throw Error(
          'This structure exceeds the browser limit of 256 atoms or 8,192 short-range neighbor pairs.',
        );
      kernel = await loadKernel(key, ...bucket);
      capacity = bucket[1];
    }
    return kernel(x, species, charge, si, sj, li, lj);
  };
}
