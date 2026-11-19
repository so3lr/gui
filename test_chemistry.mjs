import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {makePython, parseXYZ} from './web/chemistry.mjs';

const atoms = 'H 0 0 0\nF 0 0 0.917';
const molecule = parseXYZ('2\nHF\n' + atoms, 'HF');
const dynamics = readFileSync(new URL('./web/native_md.py', import.meta.url), 'utf8');

test('single-structure import rejects explicit coordinate-unit metadata', () => {
  for (const key of ['length_unit', 'coordinate_unit', 'units']) {
    assert.throws(() => parseXYZ('2\n' + key + '=bohr\n' + atoms, 'HF'), /plain XYZ/);
  }
});

test('MD scripts identify saved frames and include CUDA for automatic or GPU selection', () => {
  for (const device of ['auto', 'gpu', 'cpu']) {
    const script = makePython(molecule, 'so3lr-tauri-m', 'md', {device}, dynamics);
    assert.match(script, /frame\.info\.update\(task="md"/);
    assert.equal(script.includes('jax[cuda12]'), device !== 'cpu');
  }
});

test('Python optimization exports accept the larger step budget and tighter default force limit', () => {
  const settings = {steps: 2000};
  const model = 'so3lr-tauri-s';
  assert.match(makePython(molecule, model, 'relax', settings), /optimizer\.run\(fmax=0\.01, steps=2000\)/);
  const periodic = {...molecule, cell:[[10,0,0],[0,10,0],[0,0,10]], pbc:[true,true,true]};
  const source = readFileSync(new URL('./web/periodic_job.py', import.meta.url), 'utf8');
  assert.match(makePython(periodic, model, 'relax', settings, source), /steps=2000, fmax=0\.01/);
});
