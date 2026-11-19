import { simulate } from './browser-simulation.mjs';
self.onmessage = async ({ data: { structure, config } }) => {
  try {
    const result = await simulate(structure, config, (step, positions) =>
      postMessage({ step, positions }),
    );
    postMessage({ result });
  } catch (error) {
    postMessage({ error: error.message });
  }
};
