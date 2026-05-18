"""Per-repetition projection parameter containers for muvera-fde."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from pymuvera._internal.sketch import (
    cross_polytope_params,
    low_rank_simhash_factors,
    simhash_matrix,
    srht_params,
)


class RepParams(BaseModel):
    """Precomputed random-projection parameters for one MUVERA repetition."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cs_indices: np.ndarray | None
    cs_signs: np.ndarray | None
    simhash_mat: np.ndarray | None
    simhash_a: np.ndarray | None = None
    simhash_b: np.ndarray | None = None
    srht_d_signs: np.ndarray | None = None
    srht_sample_indices: np.ndarray | None = None
    srht_padded_dim: int | None = None
    cp_d_signs: np.ndarray | None = None
    cp_padded_dim: int | None = None


def build_rep_params(
    rep_seed: int,
    dimension: int,
    projection_dim: int,
    num_simhash_projections: int,
    use_identity: bool,
    use_low_rank_simhash: bool = False,
    simhash_rank: int = 1,
    use_srht: bool = False,
    use_cross_polytope: bool = False,
) -> RepParams:
    """Precompute projection parameters for one repetition."""
    cs_indices = cs_signs = None
    if not use_identity:
        rng = np.random.default_rng(rep_seed)
        cs_indices = rng.integers(0, projection_dim, size=dimension)
        cs_signs = 2.0 * rng.integers(0, 2, size=dimension).astype(np.float32) - 1.0

    simhash_mat = simhash_a = simhash_b = None
    srht_d_signs = srht_sample_indices = None
    srht_padded_dim = None
    cp_d_signs = None
    cp_padded_dim = None

    if use_cross_polytope:
        cp_d_signs, cp_padded_dim = cross_polytope_params(rep_seed, projection_dim)
    elif num_simhash_projections > 0:
        if use_srht:
            srht_d_signs, srht_sample_indices, srht_padded_dim = srht_params(
                rep_seed, projection_dim, num_simhash_projections
            )
        elif use_low_rank_simhash:
            simhash_a, simhash_b = low_rank_simhash_factors(
                rep_seed, projection_dim, num_simhash_projections, simhash_rank
            )
        else:
            simhash_mat = simhash_matrix(rep_seed, projection_dim, num_simhash_projections)

    return RepParams(
        cs_indices=cs_indices,
        cs_signs=cs_signs,
        simhash_mat=simhash_mat,
        simhash_a=simhash_a,
        simhash_b=simhash_b,
        srht_d_signs=srht_d_signs,
        srht_sample_indices=srht_sample_indices,
        srht_padded_dim=srht_padded_dim,
        cp_d_signs=cp_d_signs,
        cp_padded_dim=cp_padded_dim,
    )
