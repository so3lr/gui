// With the web/ directory served at TEST_URL, run this with Node and Playwright.
const assert = require('node:assert/strict');
const { chromium, webkit } = require('playwright');
const { readFile } = require('node:fs/promises');
const reference = require('./test_data/browser_forces.json');

async function main() {
  const browser =
    process.env.BROWSER === 'webkit'
      ? await webkit.launch()
      : await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const context = await browser.newContext({ reducedMotion: 'reduce' });
    const page = await context.newPage();
    const calculations = [];
    context.on('request', (req) => {
      if (req.method() !== 'GET' || new URL(req.url()).pathname.includes('/api/'))
        calculations.push(req.url());
    });
    await page.goto(process.env.TEST_URL || 'http://localhost:8000/gui/');
    await page.locator('#scrub').waitFor({ state: 'visible' });

    const selectedModel = page.locator('input[name=potential]:checked');
    assert.equal(await selectedModel.inputValue(), 'so3lr-tauri-m');
    assert.match(await page.locator('#provenance').textContent(), /Tauri M/);
    assert.equal(await page.locator('#md-steps').inputValue(), '1000');
    assert.equal(await page.locator('#duration').textContent(), '1 ps');

    for (const { id, model, e, f } of reference) {
      const actual = await page.evaluate(
        async ({ id, model }) => {
          const { createForceEvaluator } = await import('./browser-force.mjs');
          const { symbols } = await import('./atomic-data.mjs');
          const inputs = await (await fetch('molecules.json')).json();
          let structure = inputs.find((s) => s.id === id);
          if (!structure) {
            const catalog = await (await fetch('paper-catalog.json')).json();
            const item = catalog.find((s) => s.id === id);
            structure = { ...item, ...(await (await fetch(item.structure_file)).json()) };
          }
          const evaluate = createForceEvaluator(
            'so3lr-tauri-' + model,
            structure.elements.map((el) => symbols.indexOf(el)),
            structure.charge,
          );
          const { energy, forces } = await evaluate(structure.coordinates.flat());
          return { energy, forces: [...forces] };
        },
        { id, model },
      );
      assert.ok(Math.abs(e - actual.energy) < 0.002, id + ' total energy (eV)');
      assert.equal(f.length, actual.forces.length);
      assert.ok(
        f.every((v, i) => Math.abs(v - actual.forces[i]) < 0.001),
        id + ' force components (eV/Å)',
      );
    }

    const populationError = await page.evaluate(async () => {
      const { result } = await (await fetch('ala15-paper.json')).json();
      const { analyzeFrames } = await import('./browser-analysis.mjs');
      const { peptide } = await analyzeFrames({ id: 'ala15_folded' }, result.frames);
      return Math.max(
        ...Object.entries(peptide.fractions).flatMap(([code, values]) =>
          values.map((v, i) => Math.abs(result.analysis.peptide.fractions[code][i] - v)),
        ),
      );
    });
    assert.ok(populationError < 0.0001, 'Ala15 DSSP population discrepancy: ' + populationError);

    async function waitForResult() {
      await page.waitForFunction(
        () =>
          !document.querySelector('#job-error').hidden ||
          (!document.querySelector('#result').hidden &&
            document.querySelector('#provenance').textContent.includes('Browser calculation')),
        null,
        { timeout: 90_000 },
      );
      assert.ok(
        await page.locator('#job-error').isHidden(),
        await page.locator('#job-error').textContent(),
      );
    }
    await page.locator('#md-steps').fill('3');
    await page.click('#action');
    await waitForResult();
    assert.equal(await page.locator('#frame-index').textContent(), '3 fs / 3 fs');
    assert.equal(await page.locator('#scrub').getAttribute('max'), '3');
    assert.ok(await page.getByText('Secondary structure over time', { exact: true }).isVisible());

    await page.click('#import');
    await page
      .locator('#coordinates')
      .fill('3\nStretched water\nO 0 0 0\nH 1.18 0 0\nH -0.24 0.927 0');
    await page.click('#read-paste');
    assert.equal(await selectedModel.inputValue(), 'so3lr-tauri-s');
    await page.selectOption('#method', 'relax');
    await page.click('#action');
    await waitForResult();
    assert.match(await page.locator('#measurements').textContent(), /Converged/);
    const pending = page.waitForEvent('download');
    await page.click('#save-trajectory');
    const xyz = await readFile(await (await pending).path(), 'utf8');
    const energies = [...xyz.matchAll(/\benergy=([-+0-9.eE]+)/g)].map((m) => Number(m[1]));
    assert.ok(energies.length > 1 && energies.at(-1) < energies[0]);
    assert.match(xyz, /model=so3lr-tauri-s/);
    assert.match(xyz, /optimizer="FIRE"/);
    await page.click('#import');
    await page.click('#smiles-tab');
    await page.locator('#smiles-text').fill('C[NH3+]');
    await page.click('#build-smiles');
    await page.locator('#importer').waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#charge').inputValue(), '1');
    await page.selectOption('#method', 'md');
    await page.locator('#md-steps').fill('10000');
    await page.click('#action');
    await page.click('#cancel-job');
    await page.locator('#action:not(:disabled)').waitFor();
    assert.ok(await page.locator('#result').isHidden());
    await page.locator('#md-steps').fill('2');
    await page.click('#action');
    await waitForResult();
    assert.equal(await page.locator('#scrub').getAttribute('max'), '2');
    await page.goto(new URL('?example=ala15&view=structure', page.url()).href);
    await page.locator('#action:not(:disabled)').waitFor();
    assert.equal(await selectedModel.inputValue(), 'so3lr-tauri-m');
    await page.goto(new URL('?example=paper-benchmark-torsionnet500', page.url()).href);
    await page.locator('#action:not(:disabled)').waitFor();
    assert.equal(await page.locator('#force-limit').inputValue(), '0.01');
    assert.equal(await page.locator('#max-steps').inputValue(), '2000');
    assert.ok(await page.locator('#setup').evaluate(form => form.checkValidity()));
    assert.deepEqual(calculations, []);
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
