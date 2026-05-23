"""Internal random-projection primitives for muvera-fde."""

from __future__ import annotations

import numpy as np

_COUNT_SKETCH_CHUNK_SIZE: int = 1_000_000
_UINT64_MASK: np.uint64 = np.uint64(0xFFFFFFFFFFFFFFFF)

MAX_SIMHASH_PROJECTIONS: int = 30
MAX_SIMHASH_PROJECTIONS_WITH_FILL: int = 20


# ── Full-rank Gaussian SimHash ────────────────────────────────────────────


def simhash_matrix(seed: int, dimension: int, num_projections: int) -> np.ndarray:
    return (
        np.random.default_rng(seed).standard_normal((dimension, num_projections)).astype(np.float32)
    )


# ── Low-rank Gaussian SimHash ─────────────────────────────────────────────


def low_rank_simhash_factors(
    seed: int, dimension: int, num_projections: int, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) such that W ~ AB^T, A in R^{d x r}, B in R^{k x r}.

    Quality note
    ------------
    Sign-pattern agreement with full-rank Gaussian SimHash improves as r/k
    decreases. Use r/k <= 0.25 as a practical guideline (e.g. r=4, k>=16).
    Formal partition quality bounds are an open problem.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dimension, rank)).astype(np.float32)
    b = rng.standard_normal((num_projections, rank)).astype(np.float32)
    return a, b


# ── Shared Hadamard utilities ─────────────────────────────────────────────


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _fwht_batch(x: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform row-wise. Requires d = power of 2."""
    out = x.copy()
    n = out.shape[-1]
    h = 1
    while h < n:
        out = out.reshape(out.shape[0], -1, 2, h)
        u = out[:, :, 0, :].copy()
        v = out[:, :, 1, :].copy()
        out[:, :, 0, :] = u + v
        out[:, :, 1, :] = u - v
        out = out.reshape(out.shape[0], -1)
        h <<= 1
    return out


def _rademacher_and_pad(projected: np.ndarray, d_signs: np.ndarray, padded_dim: int) -> np.ndarray:
    n, d = projected.shape
    if padded_dim > d:
        padded = np.zeros((n, padded_dim), dtype=np.float32)
        padded[:, :d] = projected
    else:
        padded = projected.astype(np.float32, copy=True)
    padded *= d_signs[np.newaxis, :]
    return padded


# ── SRHT ─────────────────────────────────────────────────────────────────


def srht_params(
    seed: int, dimension: int, num_projections: int
) -> tuple[np.ndarray, np.ndarray, int]:
    padded_dim = _next_power_of_2(dimension)
    rng = np.random.default_rng(seed)
    d_signs = (2 * rng.integers(0, 2, size=padded_dim) - 1).astype(np.float32)
    sample_indices = np.sort(rng.choice(padded_dim, size=num_projections, replace=False)).astype(
        np.int64
    )
    return d_signs, sample_indices, padded_dim


def apply_srht(
    projected: np.ndarray,
    d_signs: np.ndarray,
    sample_indices: np.ndarray,
    padded_dim: int,
) -> np.ndarray:
    padded = _rademacher_and_pad(projected, d_signs, padded_dim)
    padded = _fwht_batch(padded)
    return padded[:, sample_indices]


# ── Cross-Polytope LSH ────────────────────────────────────────────────────


def cross_polytope_params(seed: int, dimension: int) -> tuple[np.ndarray, int]:
    """Return (d_signs, padded_dim) for Cross-Polytope LSH.

    Applies full SRHT rotation then argmax(|.|).
    num_partitions = 2 * padded_dim per repetition.
    """
    padded_dim = _next_power_of_2(dimension)
    rng = np.random.default_rng(seed)
    d_signs = (2 * rng.integers(0, 2, size=padded_dim) - 1).astype(np.float32)
    return d_signs, padded_dim


def apply_cross_polytope(projected: np.ndarray, d_signs: np.ndarray, padded_dim: int) -> np.ndarray:
    """Apply Cross-Polytope LSH. Returns indices in [0, 2 * padded_dim)."""
    padded = _rademacher_and_pad(projected, d_signs, padded_dim)
    rotated = _fwht_batch(padded)
    j = np.argmax(np.abs(rotated), axis=1)
    s = (rotated[np.arange(len(j)), j] > 0).astype(np.int32)
    return (2 * j + s).astype(np.int32)


# ── SimHash partition assignment (Gray-coded) ─────────────────────────────


def simhash_partition_indices(sketch_matrix: np.ndarray) -> np.ndarray:
    signs = (sketch_matrix > 0).astype(np.int32)
    binary_indices = np.zeros(signs.shape[0], dtype=np.int32)
    for j in range(signs.shape[1]):
        binary_indices = (binary_indices << 1) | signs[:, j]
    return binary_indices ^ (binary_indices >> 1)


# ── Densifying LSH fill ───────────────────────────────────────────────────


def densifying_fill(
    rep_slice: np.ndarray,
    projected: np.ndarray,
    empty_pidxs: np.ndarray,
    seed: int,
) -> None:
    """Fill empty slots via deterministic splitmix64 hash. O(num_empty).

    Inspired by Shrivastava (2014), adapted from one-permutation MinHash
    to SimHash partition fill.
    """
    n = projected.shape[0]
    if n == 0 or len(empty_pidxs) == 0:
        return
    seed_u64 = np.uint64(seed)
    for p in empty_pidxs:
        h = np.uint64(int(p)) ^ seed_u64
        h = (h + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
        h = ((h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
        h = ((h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
        h = h ^ (h >> np.uint64(31))
        rep_slice[int(p)] = projected[int(h % np.uint64(n))]


# ── Calibrated eigenbasis SimHash ────────────────────────────────────────


def calibrated_eigenbasis_simhash_matrix(
    seed: int,
    eigenvalues: np.ndarray,
    num_projections: int,
    use_eigenvalue_weighting: bool = True,
) -> np.ndarray:
    """Sample a (d, k) SimHash projection matrix in eigenbasis space.

    Parameters
    ----------
    seed:
        RNG seed for this repetition.
    eigenvalues:
        (d,) descending eigenvalues of the empirical key covariance.
    num_projections:
        Number of SimHash bits *k*.
    use_eigenvalue_weighting:
        If True (default), scale row *i* of the raw Gaussian matrix by
        ``sqrt(lambda_i)`` so that high-variance eigenbasis coordinates
        dominate the SimHash partition assignment.  This is the water-filling
        analog: the effective metric becomes the lambda-weighted inner product

            z^T diag(lambda) z'

        concentrating discrimination on the ``deff`` semantic dimensions.

        If False, use standard uniform Gaussian projection in the rotated
        space — equivalent to DEFAULT_IDENTITY applied after the eigenbasis
        rotation, useful as an ablation baseline.

    Returns
    -------
    np.ndarray
        (d, k) float32 projection matrix ready for ``projected @ W``.
    """
    d = len(eigenvalues)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, num_projections)).astype(np.float32)
    if use_eigenvalue_weighting:
        # Row i scaled by sqrt(lambda_i): contribution of coord i to sign(z@W)
        # has std proportional to lambda_i, concentrating bucket assignment on
        # high-variance eigenbasis directions.
        scale = np.sqrt(np.maximum(eigenvalues, 0.0)).astype(np.float32)
        W *= scale[:, np.newaxis]
    return W


# ── Count Sketch ──────────────────────────────────────────────────────────


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
    return values ^ (values >> np.uint64(31))


def count_sketch(input_vector: np.ndarray, final_dimension: int, seed: int) -> np.ndarray:
    """Compress via Count Sketch. Unbiased: E[<sketch(x),sketch(y)>] = <x,y>."""
    out = np.zeros(final_dimension, dtype=np.float32)
    seed_u64 = np.uint64(seed)
    sign_seed_u64 = seed_u64 ^ np.uint64(0xD6E8FEB86659FD93)
    fd_u64 = np.uint64(final_dimension)
    for start in range(0, len(input_vector), _COUNT_SKETCH_CHUNK_SIZE):
        stop = min(start + _COUNT_SKETCH_CHUNK_SIZE, len(input_vector))
        positions = np.arange(start, stop, dtype=np.uint64)
        indices = (_splitmix64(positions ^ seed_u64) % fd_u64).astype(np.intp)
        signs = np.where(
            (_splitmix64(positions ^ sign_seed_u64) & np.uint64(1)) == 0,
            np.float32(1.0),
            np.float32(-1.0),
        ).astype(np.float32)
        np.add.at(out, indices, signs * input_vector[start:stop])
    return out
