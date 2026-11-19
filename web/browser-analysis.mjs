import { alignedView } from './alignment.mjs';

// Atom order in the bundled Ac-(Ala)15-NHMe XYZ. Each Ala contributes ten
// atoms. The incomplete caps do not participate in the DSSP topology.
async function ala15Dssp(frames) {
  const { default: load } = await import('./vendor/mdtraj/dssp.mjs');
  const runtime = await load();
  const residues = 15;
  const atoms = residues * 4;
  const nco = runtime._malloc(residues * 3 * 4);
  const ca = runtime._malloc(residues * 4);
  const zeros = runtime._malloc(residues * 4);
  const xyz = runtime._malloc(atoms * 3 * 4);
  const output = runtime._malloc(residues);
  const indices = [];
  for (let i = 0; i < residues; i++) {
    const n = 6 + i * 10;
    indices.push(n, n + 2, n + 8, n + 9);
    runtime.HEAP32.set([4 * i, 4 * i + 2, 4 * i + 3], nco / 4 + 3 * i);
    runtime.HEAP32[ca / 4 + i] = 4 * i + 1;
  }
  runtime.HEAP32.fill(0, zeros / 4, zeros / 4 + residues);
  const fractions = Object.fromEntries(Array.from('HGTCBEIS', (code) => [code, []]));
  for (const frame of frames) {
    indices.forEach((atom, i) => {
      for (let k = 0; k < 3; k++) runtime.HEAPF32[xyz / 4 + 3 * i + k] = frame[atom][k] * 0.1;
    });
    runtime._dssp(xyz, nco, ca, zeros, zeros, 1, atoms, residues, output);
    const codes = String.fromCharCode(
      ...runtime.HEAPU8.subarray(output, output + residues),
    ).replaceAll(' ', 'C');
    for (const [code, values] of Object.entries(fractions)) {
      values.push(Array.from(codes).filter((value) => value === code).length / residues);
    }
  }
  for (const ptr of [nco, ca, zeros, xyz, output]) runtime._free(ptr);
  return { method: 'DSSP (MDTraj)', residue_count: residues, fractions };
}

export async function analyzeFrames(structure, frames) {
  if (structure.id === 'ala15_folded') return { peptide: await ala15Dssp(frames) };
  const heavy = structure.elements.flatMap((el, i) => (el === 'H' ? [] : [i]));
  if (!heavy.length) return {};
  const reference = frames[0];
  const rmsd = frames.map((frame) => {
    const rotated = alignedView(frame, reference, structure.elements).positions;
    const mean = [0, 0, 0];
    for (const i of heavy)
      for (let k = 0; k < 3; k++) mean[k] += (rotated[i][k] - reference[i][k]) / heavy.length;
    let squared = 0;
    for (const i of heavy)
      for (let k = 0; k < 3; k++) squared += (rotated[i][k] - reference[i][k] - mean[k]) ** 2;
    return Math.sqrt(squared / heavy.length);
  });
  return { rmsd, heavy_atom_count: heavy.length };
}
