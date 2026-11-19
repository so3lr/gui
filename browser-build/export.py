"""Lower isolated SO3LR energy and its Cartesian gradient to StableHLO."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from jax_md import space
from so3lr import So3lrPotential
from so3lr.jaxmd_utils import neighbor_list_featurizer
from so3lr.mlff.nn.stacknet import observable_function_sparse as sparse


def export(size, neighbors, model, output):
    potential = So3lrPotential(model=f"so3lr-tauri-{model}", lr_cutoff=100., dtype=jnp.float32)
    make_inputs = sparse._model_inputs

    # Keep the padded atoms in their own neutral graph so that charge
    # equilibration and molecular reductions only use the real atoms.
    def padded_inputs(positions, atomic_numbers, idx_i, idx_j, **options):
        real = atomic_numbers > 0
        options.update(
            batch_segments=(~real).astype(jnp.int32),
            node_mask=real,
            graph_mask=jnp.array([True, False]),
            total_charge=jnp.concatenate([options["total_charge"], jnp.zeros(1)]),
            theory_mask=jnp.repeat(options["theory_mask"], 2, axis=0),
            num_unpaired_electrons=jnp.zeros(2),
        )
        return make_inputs(positions, atomic_numbers, idx_i, idx_j, **options)

    def energy(positions, species, charge, short_pairs, long_pairs):
        featurize = neighbor_list_featurizer(
            space.free()[0], species, total_charge=charge, fractional_coordinates=False
        )
        return jnp.sum(jnp.where(species > 0, potential(featurize(positions, short_pairs, long_pairs)), 0))

    long_pairs = size * (size - 1) // 2
    shape = jax.ShapeDtypeStruct
    args = (shape((size, 3), jnp.float32), shape((size,), jnp.int32), shape((), jnp.float32),
            shape((2, neighbors), jnp.int32), shape((2, long_pairs), jnp.int32))
    sparse._model_inputs = padded_inputs
    try:
        output.write_text(jax.jit(jax.value_and_grad(energy)).lower(*args).as_text())
    finally:
        sparse._model_inputs = make_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", type=int)
    parser.add_argument("neighbors", type=int)
    parser.add_argument("model", choices=("s", "m", "l"))
    parser.add_argument("output", type=Path)
    export(**vars(parser.parse_args()))
