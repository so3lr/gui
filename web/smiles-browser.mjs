// Build a starting geometry from SMILES without the Python service, so the hosted
// page can accept SMILES too. OpenChemLib stands in for RDKit: it parses the
// SMILES, adds hydrogens, generates one conformer and minimises it with MMFF94.
// The checks and the returned structure follow molecule_input.from_smiles.
import { validate } from './chemistry.mjs';

const SEED = 20261119;
const VENDOR = new URL('./vendor/openchemlib/', import.meta.url);
let library;

// Loaded on first use only: the library and its parameter tables are about 2 MB.
function load() {
  library ??= import(new URL('openchemlib.js', VENDOR).href).then(async OCL => {
    await OCL.Resources.registerFromUrl(new URL('resources.json', VENDOR).href);
    return OCL;
  }).catch(error => {
    library = undefined;
    throw new Error('Could not load the SMILES builder. Check the connection and try again.', { cause: error });
  });
  return library;
}

async function digest(text) {
  const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)));
  return [...bytes].map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

// Atom indices of each disconnected fragment, in atom order.
function fragments(molecule) {
  const count = molecule.getAllAtoms(), neighbours = Array.from({ length: count }, () => []);
  for (let b = 0; b < molecule.getAllBonds(); b++) {
    const i = molecule.getBondAtom(0, b), j = molecule.getBondAtom(1, b);
    neighbours[i].push(j); neighbours[j].push(i);
  }
  const seen = new Array(count).fill(false), parts = [];
  for (let start = 0; start < count; start++) {
    if (seen[start]) continue;
    const part = [], stack = [start];
    seen[start] = true;
    while (stack.length) {
      const atom = stack.pop();
      part.push(atom);
      for (const next of neighbours[atom]) if (!seen[next]) { seen[next] = true; stack.push(next); }
    }
    parts.push(part.sort((a, b) => a - b));
  }
  return parts;
}

export async function buildInBrowser(smiles) {
  if (typeof smiles !== 'string' || !smiles.trim()) throw new Error('Enter a SMILES string, for example CCO for ethanol.');
  smiles = smiles.trim();
  if (smiles.length > 4096 || /\s/.test(smiles)) {
    throw new Error('Enter one SMILES string of up to 4096 characters, without a name or spaces.');
  }
  // OpenChemLib reads a wildcard atom as carbon, so reject it before parsing.
  if (smiles.includes('*')) throw new Error('Use elements from hydrogen to astatine. Replace any wildcard atoms.');
  const OCL = await load();
  let molecule;
  try { molecule = OCL.Molecule.fromSmiles(smiles); } catch {
    throw new Error('Could not read this SMILES. Check the rings, bonds and brackets.');
  }
  for (let i = 0; i < molecule.getAllAtoms(); i++) {
    const z = molecule.getAtomicNo(i);
    if (!(z >= 1 && z <= 85)) throw new Error('Use elements from hydrogen to astatine. Replace any wildcard atoms.');
    if (molecule.getAtomMass(i)) {
      throw new Error('Isotope-specific masses are unavailable in this workbench. Use an unlabeled structure.');
    }
  }
  const canonical = molecule.toIsomericSmiles();
  let charge = 0;
  for (let i = 0; i < molecule.getAllAtoms(); i++) charge += molecule.getAtomCharge(i);
  if (Math.abs(charge) > 10) throw new Error('The formal charge must be between −10 and +10 e.');
  molecule.addImplicitHydrogens();
  const atoms = molecule.getAllAtoms();
  if (atoms < 2 || atoms > 256) throw new Error(`Use 2–256 atoms including hydrogens. This SMILES has ${atoms}.`);

  const conformer = new OCL.ConformerGenerator(SEED).getOneConformerAsMolecule(molecule);
  if (!conformer) throw new Error('Could not generate a 3D conformer for this SMILES. Import XYZ coordinates instead.');
  let method = 'its conformer generator and up to 500 MMFF94 minimisation steps';
  try {
    new OCL.ForceFieldMMFF94(conformer, OCL.ForceFieldMMFF94.MMFF94).minimise({ maxIts: 500 });
  } catch {
    method = 'its conformer generator; MMFF94 does not cover this molecule, so the geometry is not minimised';
  }

  const elements = [], coordinates = [];
  for (let i = 0; i < conformer.getAllAtoms(); i++) {
    elements.push(conformer.getAtomLabel(i));
    coordinates.push([conformer.getAtomX(i), conformer.getAtomY(i), conformer.getAtomZ(i)]);
  }
  let source = `3D conformer generated in the browser from ${canonical} with OpenChemLib ${OCL.version}, ${method}, seed ${SEED}.`;
  const parts = fragments(conformer);
  if (parts.length > 1) {
    // Conformers of disconnected fragments can overlap. Separate their x extents,
    // retaining internal geometry and atom ordering.
    let offset = 0;
    for (const part of parts) {
      const xs = part.map(i => coordinates[i][0]), low = Math.min(...xs);
      for (const i of part) coordinates[i][0] += offset - low;
      offset = Math.max(...part.map(i => coordinates[i][0])) + 3.5;
    }
    source += ' Disconnected fragments start 3.5 Å apart along x.';
  }
  const centre = [0, 1, 2].map(k => coordinates.reduce((sum, p) => sum + p[k], 0) / coordinates.length);
  for (const p of coordinates) for (let k = 0; k < 3; k++) p[k] = Math.round((p[k] - centre[k]) * 1e8) / 1e8;
  const formula = validate({ elements, coordinates, charge });
  return {
    id: 'smiles-' + await digest(canonical),
    name: canonical.length <= 32 ? canonical : 'SMILES molecule',
    formula, charge, elements, coordinates,
    smiles: canonical,
    source,
  };
}
