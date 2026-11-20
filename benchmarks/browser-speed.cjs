// With Playwright available, pass one or more GUI URLs to compare on the same browser.
const {chromium, webkit} = require('playwright');

async function main() {
  const browser = process.env.BROWSER === 'webkit'
    ? await webkit.launch()
    : await chromium.launch({executablePath:process.env.CHROME_PATH});
  try {
    for (const url of process.argv.slice(2)) {
      const page = await browser.newPage({reducedMotion:'reduce'});
      await page.goto(url);
      const rows = await page.evaluate(async () => {
        const [core, catalog] = await Promise.all(['molecules.json', 'paper-catalog.json'].map(async path =>
          (await fetch(path)).json()));
        const results = [];
        for (const id of ['paper-test-ion','paper-benchmark-torsionnet500','ala4','ala15_folded']) {
          const item = [...core, ...catalog].find(s => s.id === id);
          const structure = item.structure_file ? {...item, ...await (await fetch(item.structure_file)).json()} : item;
          const model = structure.default_model || 'so3lr-tauri-s';
          for (const task of ['md', 'relax']) {
            const worker = new Worker('./simulation-worker.mjs', {type:'module'});
            try {
              const elapsed = await new Promise((resolve, reject) => {
                let start, last, step;
                worker.onerror = e => reject(Error(e.message));
                worker.onmessage = ({data}) => {
                  if (data.error) return reject(Error(data.error));
                  if (data.result) {
                    if (step <= 2) return reject(Error('Run ended during warm-up: ' + id));
                    return resolve((last - start) / (step - 2));
                  }
                  step = data.step;
                  last = performance.now();
                  if (step === 2) start = last;
                };
                worker.postMessage({structure,config:{
                  task,model,charge:structure.charge,steps:24,fmax:0.01,seed:51,timestep:1,temperature:300,
                }});
              });
              results.push({id,model,task,msPerStep:Number(elapsed.toFixed(2))});
            } finally {
              worker.terminate();
            }
          }
        }
        return results;
      });
      console.log(JSON.stringify({url,rows}));
      await page.close();
    }
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
