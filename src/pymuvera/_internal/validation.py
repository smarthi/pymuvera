"""Validation helpers for FDEConfig."""

from __future__ import annotations

import numpy as np

from pymuvera._internal.sketch import (
    MAX_SIMHASH_PROJECTIONS,
    MAX_SIMHASH_PROJECTIONS_WITH_FILL,
    _next_power_of_2,
)
from pymuvera.config import FDEConfig, ProjectionType

_MAX_INTERMEDIATE_FDE_BYTES: int = 1 << 30


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_simhash_projections(config: FDEConfig) -> None:
    if config.projection_type == ProjectionType.CROSS_POLYTOPE:
        return  # k is ignored for CROSS_POLYTOPE
    limit = (
        MAX_SIMHASH_PROJECTIONS_WITH_FILL
        if config.fill_empty_partitions
        else MAX_SIMHASH_PROJECTIONS
    )
    if not (0 <= config.num_simhash_projections <= limit):
        suffix = " when fill_empty_partitions=True" if config.fill_empty_partitions else ""
        raise ValueError(
            f"num_simhash_projections must be in [0, {limit}]{suffix}, "
            f"got {config.num_simhash_projections}"
        )


def _check_projection_dimension(config: FDEConfig) -> None:
    if config.projection_type == ProjectionType.AMS_SKETCH:
        if config.projection_dimension is None or config.projection_dimension <= 0:
            raise ValueError("A positive projection_dimension must be set when using AMS_SKETCH.")


def _check_simhash_rank(config: FDEConfig) -> None:
    if config.projection_type != ProjectionType.LOW_RANK_GAUSSIAN:
        return
    if config.simhash_rank <= 0:
        raise ValueError(
            f"simhash_rank must be positive for LOW_RANK_GAUSSIAN, got {config.simhash_rank}"
        )
    if config.num_simhash_projections > 0 and config.simhash_rank >= config.num_simhash_projections:
        raise ValueError(
            f"simhash_rank ({config.simhash_rank}) must be strictly less than "
            f"num_simhash_projections ({config.num_simhash_projections})."
        )


def _check_srht(config: FDEConfig) -> None:
    if config.projection_type != ProjectionType.SRHT:
        return
    if config.num_simhash_projections == 0:
        return
    proj_dim = (
        config.projection_dimension
        if config.projection_type == ProjectionType.AMS_SKETCH
        else config.dimension
    )
    padded_dim = _next_power_of_2(proj_dim)
    if config.num_simhash_projections > padded_dim:
        raise ValueError(
            f"SRHT requires num_simhash_projections ({config.num_simhash_projections}) "
            f"<= next_power_of_2(dimension) = {padded_dim}."
        )


def _check_calibrated_eigenbasis(config: FDEConfig) -> None:
    if config.projection_type != ProjectionType.CALIBRATED_EIGENBASIS:
        return
    if config.num_simhash_projections < 1:
        raise ValueError(
            "CALIBRATED_EIGENBASIS requires num_simhash_projections >= 1, "
            f"got {config.num_simhash_projections}"
        )


def validate_config(config: FDEConfig) -> None:
    _check_positive(config.dimension, "dimension")
    _check_positive(config.num_repetitions, "num_repetitions")
    if config.final_projection_dimension is not None:
        _check_positive(config.final_projection_dimension, "final_projection_dimension")
    _check_simhash_projections(config)
    _check_projection_dimension(config)
    _check_simhash_rank(config)
    _check_srht(config)
    _check_calibrated_eigenbasis(config)


def num_partitions_for_config(config: FDEConfig, projection_dim: int) -> int:
    """Return the number of partitions per repetition for this config."""
    if config.projection_type == ProjectionType.CROSS_POLYTOPE:
        from pymuvera._internal.sketch import _next_power_of_2 as _np2

        return 2 * _np2(projection_dim)
    return 1 << config.num_simhash_projections


def prepare_embeddings(point_cloud: np.ndarray, config: FDEConfig) -> np.ndarray:
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.ndim == 2:
        if point_cloud.shape[1] != config.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: got {point_cloud.shape[1]}, "
                f"expected {config.dimension}"
            )
        return point_cloud
    if point_cloud.ndim == 1:
        if len(point_cloud) % config.dimension != 0:
            raise ValueError(
                f"Flat point-cloud length {len(point_cloud)} not divisible by "
                f"dimension {config.dimension}"
            )
        return point_cloud.reshape(-1, config.dimension)
    raise ValueError(f"point_cloud must be 1-D or 2-D, got {point_cloud.ndim}-D")


def checked_intermediate_fde_length(
    config: FDEConfig, projection_dim: int, num_partitions: int | None = None
) -> int:
    if num_partitions is None:
        num_partitions = num_partitions_for_config(config, projection_dim)
    fde_length = config.num_repetitions * num_partitions * projection_dim
    required_bytes = fde_length * np.dtype(np.float32).itemsize
    if required_bytes > _MAX_INTERMEDIATE_FDE_BYTES:
        raise ValueError(
            f"Configuration would allocate {required_bytes} bytes intermediate FDE. "
            "Reduce num_simhash_projections, num_repetitions, or projection_dimension."
        )
    return fde_length
