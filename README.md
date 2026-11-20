# SO3LR

Optimize a molecule, run molecular dynamics, or explore a saved trajectory in your browser. The opening view plays a 500 ps Ala15 recording. To start your own calculation, open an XYZ file, build a structure from SMILES, or choose an example. [Open the SO3LR GUI](https://so3lr.github.io/gui/)

## Start the workbench

From the directory containing this README, run the following command. On first launch, uv downloads the pinned public dependencies into an isolated environment. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv run --locked serve.py --port 8773
```

Open the workbench in your browser. The first view plays the Ala15 recording. Playback starts paused when the browser requests reduced motion. During optimization and molecular dynamics, the viewer follows the latest saved geometry and shows its step. Periodic optimization updates the unit cell too. The local server can reconnect to an active calculation after a page refresh and supports trajectory import and periodic cell relaxation. A browser-only calculation runs while its tab stays open. [Open SO3LR](http://127.0.0.1:8773/)

## Explore the 500 ps Ala15 trajectory

The opening recording shows the 162-atom capped peptide Ac-(Ala)₁₅-NHMe in the gas phase at a target of 300 K. It comes from replica 1 (run `r2`) of the Tauri paper’s folded Ala15 study. The original SO3LR-Tauri M production run used JAX-MD with a Nosé–Hoover chain and a 0.5 fs integration step. The replay covers the first 500 ps in 501 snapshots, starting at 0.5 ps and then every 1 ps. The viewer aligns heavy atoms to the first frame to remove overall tumbling. Exported coordinates retain their original orientation. [Tauri, Figure S16](https://arxiv.org/html/2611.06237v1) · [Recording preparation](benchmarks/prepare_ala15_recording.py)

Playback advances at 30 saved frames per second at 1×. Use the speed selector beside Play to slow down or speed up any trajectory, or drag the slider to inspect a frame. Clicking or dragging in an analysis chart moves the structure and the other chart cursors to the same frame. Use the arrow keys when a chart has keyboard focus. Choose **Restart recording** to return to the first frame, or **Open 500 ps recording** to switch back from a calculation on Ala15.

- **Secondary structure** reports the fraction of all 15 alanine residues assigned to each DSSP class at every displayed time. Legend percentages average over all saved frames and all 15 residues. The chart follows the population at each frame. Caps are excluded. Toggle α-helix, 3₁₀-helix, turn, coil or another observed state in the legend. The GUI computes DSSP from the coordinates. The paper used STRIDE. Transitions faster than the display sampling interval can fall between snapshots. Peptide-specific analysis appears for Ala15, including recognized imported trajectories. [MDTraj DSSP](https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html)
- **Kinetic energy** and **instantaneous temperature** use the archived momenta and the paper run’s 3N temperature convention. The two raw trajectory segments omit potential energy, so the opening replay has no potential-energy curve.
- **Save trajectory** exports the displayed snapshots as extended XYZ. The replay’s timestamps preserve the original sampling times.

## Find an example

Choose **Examples** above the viewer. Each card shows the structure name, its source, and the atom count. Choose **Points** in the viewer to see atoms as colored points.

## Enter your own structure

- **Open XYZ** accepts an XYZ file or pasted coordinates in ångströms. An integer `charge=` in the comment sets the molecular charge.
- **Build from SMILES** accepts strings such as `CCO` and `C[NH3+]`. The browser uses OpenChemLib to build a seeded conformer and run up to 500 MMFF94 minimization steps. The local service uses RDKit to preserve specified stereochemistry and formal charge, add hydrogens, and generate a seeded ETKDGv3 conformer. Where parameters are available, it refines the geometry using MMFF94 or UFF. For disconnected inputs, the local builder places fragments 3.5 Å apart along x. [OpenChemLib](https://github.com/cheminfo/openchemlib-js)
- **Open trajectory** imports consistently ordered XYZ or extended XYZ frames through the local service. Fully periodic frames use three lattice vectors and three periodic axes. For molecular MD without `time_fs`, enter the time between saved frames. Imports support up to 5,000 frames and 500,000 atom positions, with up to 256 atoms in an isolated frame or 50,000 in a periodic frame. Charts use the available data. Coordinates suffice for molecular RMSD, lattice vectors for cell volume, and archived kinetic energies or momenta for kinetic energy.

## Run a calculation

Choose a calculation type and Tauri model. Press **Optimize geometry** or **Run dynamics** to start from the input geometry. **Download Python job** saves a script with the input and settings. To return to a previous result, open **Run history**.

The browser runs isolated structures of 2–256 atoms and up to 8,192 directed short-range neighbor pairs on the device’s CPU. It uses L-BFGS optimization or BAOAB Langevin dynamics, with float32 energies and forces. Large or dense structures take longer. The first run with a model downloads its files from the site. Input coordinates stay on the device. [Runtime build](browser-build/README.md) · [Model attribution and licenses](web/browser-runtime/NOTICE.txt)

The optional local service queues one job at a time and accepts 2–256 atoms. Isolated systems support L-BFGS geometry optimization and Langevin MD with integer total charge from −10 to +10 e. Neutral three-dimensionally periodic cells support relaxation of both atom positions and cell vectors using LBFGS. Periodic calculations use exact real-space neighbors with a 12 Å cutoff and particle-mesh Ewald electrostatics. Isolated calculations use a 100 Å long-range cutoff. Energies are in eV and forces in eV/Å. [SO3LR ASE calculator](https://so3lr.github.io/usage/ase/)

For **Geometry optimization**, the default force tolerance is 0.01 eV/Å. For isolated molecules, convergence means every atom’s force magnitude is at or below the threshold. Benzimidazole–indazole has a budget of 2,000 steps. Its starting geometry is the original twisted benchmark conformer. Other structures start with 100 steps, and the budget is adjustable up to 2,000. The optimizer stops early when it converges.

For **Molecular dynamics**, the default is 1,000 steps of 1.0 fs at 300 K. Each new MD run draws an integer seed from 0 through 100. The saved MD trajectory includes the seed used. The local service and the downloaded Python job use the same JAX BAOAB Langevin integrator with friction 0.01 fs⁻¹. A fixed center of mass leaves 3N−3 thermal degrees of freedom for live dynamics. Jobs of up to 2,000 steps retain up to 101 frames and longer jobs retain up to 1,001. The local service stops at an integration block boundary. Cancelling a browser job terminates its worker and restores the input structure.

Live MD results include potential energy, kinetic energy, instantaneous temperature and heavy-atom RMSD. **Save trajectory** beside the playback controls writes available frame properties. The downloaded Python job can stream snapshots to disk.

## Optional CUDA support

Exported Python jobs with Automatic or GPU device selection include CUDA 12 dependencies on Linux x86-64. For the local GUI on Linux x86-64 with a compatible NVIDIA driver, use the CUDA 12 extra. Browser and server can run on different computers through SSH port forwarding. The server host performs the calculation. [JAX installation](https://docs.jax.dev/en/latest/installation.html)

```sh
uv run --locked --extra cuda12 serve.py --port 8773
```

## Sources and attribution

The core molecular inputs include aspirin imported from the SMILES `CC(=O)Oc1ccccc1C(=O)O` (PubChem CID 2244; the same seeded conformer is bundled for offline use), the stacked JSCH-71 guanine–cytosine pair, the first MD22 frame of docosahexaenoic acid, and two capped peptides. **Ala4** is the MD22 42-atom Ac-(Ala)₃-NHMe example with 100 conformers from the SO3LR test file. Use **Source XYZ** for the original peptide files. [MD22 data](https://www.sgdml.org/) · [MD22 paper](https://doi.org/10.1126/sciadv.adf0873) · [Ala4 source](https://github.com/general-molecular-simulations/so3lr/blob/e13d51b094bd7e2e7b8a4758aacdbafea9e82819/tests/test_data/ac.xyz) · [Ala15 source](https://github.com/general-molecular-simulations/so3lr/blob/e13d51b094bd7e2e7b8a4758aacdbafea9e82819/examples/data/ala15_folded.xyz)

The browser SMILES builder uses OpenChemLib 9.25.1 (BSD-3-Clause), bundled with its licence in `web/vendor/openchemlib/`; `resources.json` keeps only its `cod/` and `forcefield/` entries. [OpenChemLib](https://github.com/cheminfo/openchemlib-js)

Kabylda and coauthors released the Tauri paper data under CC BY 4.0. The library includes representative geometries for larger benchmark collections. The catalog identifies the selected archive members and frames. Additional upstream attributions are listed in the coverage notes. The header and favicon use the orbital mark from the SO3LR documentation. [Zenodo release](https://doi.org/10.5281/zenodo.22920714) · [Molecular benchmark catalog](web/paper-catalog.json) · [Coverage and upstream credits](benchmarks/paper-examples.md) · [SO3LR documentation](https://so3lr.github.io/)

The model is pinned to SO3LR 0.2.0 at commit `e13d51b094bd7e2e7b8a4758aacdbafea9e82819`. Upstream code is MIT-licensed. The Tauri model weights are licensed under CC BY-NC 4.0. [SO3LR source](https://github.com/general-molecular-simulations/so3lr/tree/e13d51b094bd7e2e7b8a4758aacdbafea9e82819) · [Upstream notices](https://github.com/general-molecular-simulations/so3lr/blob/e13d51b094bd7e2e7b8a4758aacdbafea9e82819/NOTICES.md) · [SO3LR Tauri paper and supporting information](https://arxiv.org/html/2611.06237v1)
