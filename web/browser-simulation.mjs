import { validate } from './chemistry.mjs';
import { analyzeFrames } from './browser-analysis.mjs';
import { createForceEvaluator } from './browser-force.mjs';
import { symbols, masses } from './atomic-data.mjs';

const fsInAseUnits = 0.09822694788464063;
const kB = 8.617330337217213e-5;
const triples = (v) =>
  Array.from({ length: v.length / 3 }, (_, i) => Array.from(v.slice(3 * i, 3 * i + 3)));
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
function maxNorm(x) {
  let max = 0;
  for (let i = 0; i < x.length; i += 3) max = Math.max(max, Math.hypot(x[i], x[i + 1], x[i + 2]));
  return max;
}
function fireStep(velocity) {
  let dt = 0.1, alpha = 0.1, positive = 0;
  return (x, f, step) => {
    if (dot(velocity, f) > 0) {
      const ratio = Math.sqrt(dot(velocity, velocity) / dot(f, f));
      for (let i = 0; i < x.length; i++)
        velocity[i] = (1 - alpha) * velocity[i] + alpha * ratio * f[i];
      if (++positive > 5) {
        dt = Math.min(1, 1.1 * dt);
        alpha *= 0.99;
      }
    } else if (step) {
      velocity.fill(0);
      dt *= 0.5;
      alpha = 0.1;
      positive = 0;
    }
    for (let i = 0; i < x.length; i++) velocity[i] += dt * f[i];
    const scale = Math.min(dt, 0.15 / Math.max(maxNorm(velocity), 1e-30));
    for (let i = 0; i < x.length; i++) x[i] += scale * velocity[i];
  };
}

// Limited-memory BFGS in Cartesian coordinates. The two-loop recursion
// applies an inverse Hessian approximation to the negative energy gradient.
function lbfgsStep() {
  const history = [];
  let displacement, previousForces;
  return (positions, forces) => {
    if (previousForces) {
      const gradientChange = previousForces.map((v, i) => v - forces[i]);
      const curvature = dot(displacement, gradientChange);
      if (curvature > 1e-12) {
        history.push({ displacement, gradientChange, inverseCurvature: 1 / curvature });
        if (history.length > 20) history.shift();
      }
    }
    const direction = forces.slice();
    const alpha = [];
    for (let i = history.length - 1; i >= 0; i--) {
      const pair = history[i];
      alpha[i] = dot(pair.displacement, direction) * pair.inverseCurvature;
      for (let k = 0; k < direction.length; k++) direction[k] -= alpha[i] * pair.gradientChange[k];
    }
    // Initial diagonal stiffness in eV/Å², as in ASE LBFGS.
    for (let k = 0; k < direction.length; k++) direction[k] /= 70;
    for (let i = 0; i < history.length; i++) {
      const pair = history[i];
      const beta = dot(pair.gradientChange, direction) * pair.inverseCurvature;
      for (let k = 0; k < direction.length; k++) direction[k] += (alpha[i] - beta) * pair.displacement[k];
    }
    const scale = Math.min(1, 0.15 / maxNorm(direction));
    displacement = direction.map(v => v * scale);
    for (let i = 0; i < positions.length; i++) positions[i] += displacement[i];
    previousForces = forces;
  };
}

function normals(seed) {
  // Mulberry32 with Box–Muller normal sampling.
  function uniform() {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  }
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

export async function simulate(structure, config, onFrame) {
  if (structure.cell) throw Error('Periodic cells need the local engine.');
  validate({ ...structure, charge: config.charge });
  const z = structure.elements.map((el) => symbols.indexOf(el));
  const mass = z.map((i) => masses[i]),
    massSum = mass.reduce((a, b) => a + b, 0);
  const x = Float64Array.from(structure.coordinates.flat()),
    velocity = new Float64Array(x.length);
  const evaluate = createForceEvaluator(config.model, z, config.charge);
  const result = {
    frames: [],
    frame_steps: [],
    energies: [],
    model: config.model,
    precision: 'float32',
  };
  let step = 0,
    state = await evaluate(x);
  function record(ke) {
    const frame = triples(x);
    result.frames.push(frame);
    result.frame_steps.push(step);
    result.energies.push(state.energy);
    if (ke !== undefined) {
      result.times_fs.push(step * config.timestep);
      result.kinetic_energies.push(ke);
      result.temperatures.push((2 * ke) / ((x.length - 3) * kB));
    }
    onFrame(step, frame);
  }
  if (config.task === 'md') {
    const dt = config.timestep * fsInAseUnits,
      decay = Math.exp(-0.01 * config.timestep);
    const noiseScale = Math.sqrt(-Math.expm1(-0.02 * config.timestep));
    const gaussian = normals(config.seed),
      thermal = mass.map((m) => Math.sqrt((kB * config.temperature) / m));
    const noise = new Float64Array(x.length);
    function removeCOM(v) {
      for (let k = 0; k < 3; k++) {
        let mean = 0;
        for (let i = 0; i < mass.length; i++) mean += (v[3 * i + k] * mass[i]) / massSum;
        for (let i = k; i < v.length; i += 3) v[i] -= mean;
      }
    }
    const kinetic = () =>
      velocity.reduce((e, v, i) => e + 0.5 * mass[Math.floor(i / 3)] * v * v, 0);
    function draw(v) {
      for (let i = 0; i < v.length; i++) v[i] = thermal[Math.floor(i / 3)] * gaussian();
      removeCOM(v);
    }
    function kick() {
      for (let i = 0; i < x.length; i++)
        velocity[i] += (0.5 * dt * state.forces[i]) / mass[Math.floor(i / 3)];
      removeCOM(velocity);
    }
    Object.assign(result, {
      times_fs: [],
      temperatures: [],
      kinetic_energies: [],
      md: {
        integrator: 'BAOAB Langevin',
        engine: 'wasm',
        ensemble: 'NVT',
        thermostat: 'Langevin',
        friction_per_fs: 0.01,
        temperature: config.temperature,
        temperature_dof: x.length - 3,
      },
    });
    draw(velocity);
    record(kinetic());
    const stride = Math.max(1, Math.ceil(config.steps / (config.steps > 2000 ? 1000 : 100)));
    for (step = 1; step <= config.steps; step++) {
      kick();
      draw(noise);
      for (let i = 0; i < x.length; i++) {
        x[i] += 0.5 * dt * velocity[i];
        velocity[i] = decay * velocity[i] + noiseScale * noise[i];
        x[i] += 0.5 * dt * velocity[i];
      }
      state = await evaluate(x);
      kick();
      if (step % stride === 0 || step === config.steps) record(kinetic());
    }
  } else {
    const optimizer = structure.browser_optimizer === 'LBFGS' ? 'LBFGS' : 'FIRE';
    const advance = optimizer === 'LBFGS' ? lbfgsStep() : fireStep(velocity);
    record();
    while (step < config.steps && maxNorm(state.forces) > config.fmax) {
      advance(x, state.forces, step);
      state = await evaluate(x);
      step++;
      record();
    }
    result.converged = maxNorm(state.forces) <= config.fmax;
    result.optimizer = optimizer;
  }
  result.coordinates = triples(x);
  result.energy = state.energy;
  result.forces = triples(state.forces);
  result.analysis = await analyzeFrames(structure, result.frames);
  return result;
}
