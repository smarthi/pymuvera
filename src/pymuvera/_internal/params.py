"""Per-repetition projection parameter containers for pymuvera."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from pymuvera._internal.sketch import (
    calibrated_eigenbasis_simhash_matrix,
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
    # CALIBRATED_EIGENBASIS fields
    eigenbasis_rotation: np.ndarray | None = None  # (d, d) U matrix
    eigenbasis_simhash_mat: np.ndarray | None = None  # (d, k) in eigenbasis space


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
    use_calibrated_eigenbasis: bool = False,
    calibration_eigenvalues: np.ndarray | None = None,
    calibration_eigenvectors: np.ndarray | None = None,
    use_eigenvalue_weighting: bool = True,
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
    eigenbasis_rotation = None
    eigenbasis_simhash_mat = None

    if use_calibrated_eigenbasis:
        assert calibration_eigenvalues is not None and calibration_eigenvectors is not None, (
            "calibration_eigenvalues and calibration_eigenvectors must be provided "
            "when use_calibrated_eigenbasis=True"
        )
        eigenbasis_rotation = calibration_eigenvectors  # (d, d)
        eigenbasis_simhash_mat = calibrated_eigenbasis_simhash_matrix(
            rep_seed,
            calibration_eigenvalues,
            num_simhash_projections,
            use_eigenvalue_weighting=use_eigenvalue_weighting,
        )
    elif use_cross_polytope:
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
        eigenbasis_rotation=eigenbasis_rotation,
        eigenbasis_simhash_mat=eigenbasis_simhash_mat,
    )
