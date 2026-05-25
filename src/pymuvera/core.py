"""Core FDE generation for pymuvera."""

from __future__ import annotations

import numpy as np

from pymuvera._internal.params import RepParams, build_rep_params
from pymuvera._internal.sketch import (
    apply_cross_polytope,
    apply_srht,
    count_sketch,
    densifying_fill,
    simhash_partition_indices,
)
from pymuvera._internal.validation import (
    checked_intermediate_fde_length,
    num_partitions_for_config,
    prepare_embeddings,
    validate_config,
)
from pymuvera.config import FDEConfig, ProjectionType

# ── Config-derived helpers ────────────────────────────────────────────────


def _projection_dim_for(config: FDEConfig) -> int:
    if config.projection_type == ProjectionType.AMS_SKETCH:
        assert config.projection_dimension is not None
        return config.projection_dimension
    return config.dimension


def _use_identity(config: FDEConfig) -> bool:
    return config.projection_type != ProjectionType.AMS_SKETCH


def _use_low_rank_simhash(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.LOW_RANK_GAUSSIAN


def _use_srht(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.SRHT


def _use_cross_polytope(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.CROSS_POLYTOPE


def _use_calibrated_eigenbasis(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.CALIBRATED_EIGENBASIS


def _use_densifying(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.CROSS_POLYTOPE or config.densifying_fill


# ── Projection + partition ────────────────────────────────────────────────


def _project_and_partition(
    embedding_matrix: np.ndarray,
    rep_params: RepParams,
    use_identity: bool,
    projection_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    num_points = embedding_matrix.shape[0]

    sketch_matrix: np.ndarray | None

    # CALIBRATED_EIGENBASIS: rotate into eigenbasis first, then project with
    # the eigenvalue-weighted SimHash matrix.  The stored FDE centroid lives in
    # the eigenbasis space; inner products are preserved exactly (U is orthogonal).
    if rep_params.eigenbasis_rotation is not None:
        assert rep_params.eigenbasis_simhash_mat is not None
        projected = embedding_matrix @ rep_params.eigenbasis_rotation  # (N, d) in eigenbasis
        sketch_matrix = projected @ rep_params.eigenbasis_simhash_mat  # (N, k)
        partition_indices = simhash_partition_indices(sketch_matrix)
        return projected, partition_indices, sketch_matrix

    if use_identity:
        projected = embedding_matrix
    else:
        assert rep_params.cs_indices is not None and rep_params.cs_signs is not None
        projected = np.zeros((num_points, projection_dim), dtype=np.float32)
        np.add.at(projected.T, rep_params.cs_indices, (embedding_matrix * rep_params.cs_signs).T)

    if rep_params.cp_d_signs is not None:
        # Cross-Polytope: full rotation + argmax
        assert rep_params.cp_padded_dim is not None
        partition_indices = apply_cross_polytope(
            projected, rep_params.cp_d_signs, rep_params.cp_padded_dim
        )
        sketch_matrix = None
    elif rep_params.simhash_mat is not None:
        sketch_matrix = projected @ rep_params.simhash_mat
        partition_indices = simhash_partition_indices(sketch_matrix)
    elif rep_params.simhash_a is not None:
        assert rep_params.simhash_b is not None
        sketch_matrix = (projected @ rep_params.simhash_a) @ rep_params.simhash_b.T
        partition_indices = simhash_partition_indices(sketch_matrix)
    elif rep_params.srht_d_signs is not None:
        assert rep_params.srht_sample_indices is not None
        assert rep_params.srht_padded_dim is not None
        sketch_matrix = apply_srht(
            projected,
            rep_params.srht_d_signs,
            rep_params.srht_sample_indices,
            rep_params.srht_padded_dim,
        )
        partition_indices = simhash_partition_indices(sketch_matrix)
    else:
        sketch_matrix = None
        partition_indices = np.zeros(num_points, dtype=np.int32)

    return projected, partition_indices, sketch_matrix


# ── Hamming NN empty-partition fill ──────────────────────────────────────


def _fill_empty_partitions_hamming(
    rep_slice: np.ndarray,
    projected: np.ndarray,
    empty_pidxs: np.ndarray,
    signs_rev: np.ndarray,
    k: int,
) -> None:
    empty_binary = empty_pidxs.copy()
    empty_binary ^= empty_binary >> 1
    empty_binary ^= empty_binary >> 2
    empty_binary ^= empty_binary >> 4
    empty_binary ^= empty_binary >> 8
    empty_binary ^= empty_binary >> 16

    bit_positions = np.arange(k)[np.newaxis, :]
    num_points = signs_rev.shape[0]
    if num_points == 0:
        return

    batch_size = max(1, (1 << 20) // max(1, num_points))
    for start in range(0, len(empty_pidxs), batch_size):
        stop = min(start + batch_size, len(empty_pidxs))
        batch_pidxs = empty_pidxs[start:stop]
        batch_binary = empty_binary[start:stop]
        batch_bits = ((batch_binary[:, np.newaxis] >> bit_positions) & 1).astype(np.int32)
        batch_distances = (batch_bits[:, np.newaxis, :] != signs_rev[np.newaxis, :, :]).sum(axis=2)
        nearest = np.argmin(batch_distances, axis=1)
        rep_slice[batch_pidxs] = projected[nearest]


def _normalize_and_fill_rep(
    rep_slice: np.ndarray,
    partition_indices: np.ndarray,
    projected: np.ndarray,
    sketch_matrix: np.ndarray | None,
    config: FDEConfig,
    num_partitions: int,
    rep_seed: int,
) -> None:
    partition_sizes = np.bincount(partition_indices, minlength=num_partitions).astype(np.float32)
    filled_mask = partition_sizes > 0
    rep_slice[filled_mask] /= partition_sizes[filled_mask, np.newaxis]

    if config.fill_empty_partitions:
        empty_pidxs = np.nonzero(~filled_mask)[0]
        if len(empty_pidxs) > 0:
            if _use_densifying(config):
                densifying_fill(rep_slice, projected, empty_pidxs, rep_seed)
            elif sketch_matrix is not None:
                signs_rev = (sketch_matrix[:, ::-1] > 0).astype(np.int32)
                _fill_empty_partitions_hamming(
                    rep_slice, projected, empty_pidxs, signs_rev, config.num_simhash_projections
                )


def _maybe_count_sketch(out: np.ndarray, config: FDEConfig) -> np.ndarray:
    if config.final_projection_dimension is not None:
        return count_sketch(out, config.final_projection_dimension, config.seed)
    return out


# ── Public generation functions ───────────────────────────────────────────


def generate_query_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: list[RepParams] | None = None,
    calibration_eigenvalues: np.ndarray | None = None,
    calibration_eigenvectors: np.ndarray | None = None,
) -> np.ndarray:
    validate_config(config)
    if config.fill_empty_partitions:
        raise ValueError("Query FDE does not support fill_empty_partitions.")

    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_id = _use_identity(config)
    projection_dim = _projection_dim_for(config)
    num_partitions = num_partitions_for_config(config, projection_dim)

    out = np.zeros(
        checked_intermediate_fde_length(config, projection_dim, num_partitions),
        dtype=np.float32,
    )

    use_cb = _use_calibrated_eigenbasis(config)

    for rep in range(config.num_repetitions):
        params = (
            rep_params_list[rep]
            if rep_params_list is not None
            else build_rep_params(
                config.seed + rep,
                config.dimension,
                projection_dim,
                config.num_simhash_projections,
                use_id,
                use_low_rank_simhash=_use_low_rank_simhash(config),
                simhash_rank=config.simhash_rank,
                use_srht=_use_srht(config),
                use_cross_polytope=_use_cross_polytope(config),
                use_calibrated_eigenbasis=use_cb,
                calibration_eigenvalues=calibration_eigenvalues,
                calibration_eigenvectors=calibration_eigenvectors,
                use_eigenvalue_weighting=config.use_eigenvalue_weighting,
            )
        )
        projected, partition_indices, _ = _project_and_partition(
            embedding_matrix, params, use_id, projection_dim
        )
        rep_offset = rep * num_partitions * projection_dim
        rep_slice = out[rep_offset : rep_offset + num_partitions * projection_dim].reshape(
            num_partitions, projection_dim
        )
        np.add.at(rep_slice, partition_indices, projected)

    return _maybe_count_sketch(out, config)


def generate_document_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: list[RepParams] | None = None,
    calibration_eigenvalues: np.ndarray | None = None,
    calibration_eigenvectors: np.ndarray | None = None,
) -> np.ndarray:
    validate_config(config)
    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_id = _use_identity(config)
    projection_dim = _projection_dim_for(config)
    num_partitions = num_partitions_for_config(config, projection_dim)

    out = np.zeros(
        checked_intermediate_fde_length(config, projection_dim, num_partitions),
        dtype=np.float32,
    )

    use_cb = _use_calibrated_eigenbasis(config)

    for rep in range(config.num_repetitions):
        rep_seed = config.seed + rep
        params = (
            rep_params_list[rep]
            if rep_params_list is not None
            else build_rep_params(
                rep_seed,
                config.dimension,
                projection_dim,
                config.num_simhash_projections,
                use_id,
                use_low_rank_simhash=_use_low_rank_simhash(config),
                simhash_rank=config.simhash_rank,
                use_srht=_use_srht(config),
                use_cross_polytope=_use_cross_polytope(config),
                use_calibrated_eigenbasis=use_cb,
                calibration_eigenvalues=calibration_eigenvalues,
                calibration_eigenvectors=calibration_eigenvectors,
                use_eigenvalue_weighting=config.use_eigenvalue_weighting,
            )
        )
        projected, partition_indices, sketch_matrix = _project_and_partition(
            embedding_matrix, params, use_id, projection_dim
        )
        rep_offset = rep * num_partitions * projection_dim
        rep_slice = out[rep_offset : rep_offset + num_partitions * projection_dim].reshape(
            num_partitions, projection_dim
        )
        np.add.at(rep_slice, partition_indices, projected)
        _normalize_and_fill_rep(
            rep_slice,
            partition_indices,
            projected,
            sketch_matrix,
            config,
            num_partitions,
            rep_seed,
        )

    return _maybe_count_sketch(out, config)
