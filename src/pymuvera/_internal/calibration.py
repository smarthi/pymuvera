"""Eigenbasis calibration for CALIBRATED_EIGENBASIS projection mode.

Computes the per-head key covariance eigenbasis from a calibration batch of
embeddings.  The calibration object is the only external state that
CALIBRATED_EIGENBASIS needs beyond the shared FDEConfig.

SpectralQuant (Vangara & Gopinath, 2026) demonstrates that LLM key vectors
are strongly non-isotropic: the participation ratio deff/dh ≈ 3–5% at dh=128
across Qwen2.5, Mistral-7B, and Llama-3.  The same structure is expected in
ColQwen2 / ColQwen3.5 patch embeddings since they are produced by trained
attention heads operating on the same class of transformer architecture.

The participation ratio is defined as:
    deff = (Σ λ_i)² / Σ λ_i²

A uniform spectrum gives deff = d; single-direction concentration gives deff = 1.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class EigenbasisCalibration:
    """Precomputed eigenbasis for CALIBRATED_EIGENBASIS SimHash rotation.

    Attributes
    ----------
    eigenvectors:
        (d, d) orthonormal matrix whose columns are the eigenvectors of the
        empirical key covariance, sorted in descending eigenvalue order.
        ``x @ eigenvectors`` maps embeddings into the eigenbasis.
    eigenvalues:
        (d,) non-negative eigenvalues of the covariance, descending.
    participation_ratio:
        Soft effective rank: (Σ λ_i)² / Σ λ_i².  For ColQwen2 at d=128 this
        is expected to lie in [4, 8] based on SpectralQuant measurements on
        comparable transformer architectures.
    effective_rank:
        ``participation_ratio`` rounded to the nearest integer.
    n_samples:
        Number of token embeddings used to compute the calibration.
    """

    eigenvectors: np.ndarray  # (d, d) float32
    eigenvalues: np.ndarray  # (d,)   float32, descending
    participation_ratio: float
    effective_rank: int
    n_samples: int

    def save(self, path: str | Path) -> None:
        """Serialise to a compressed .npz file.

        Usage::

            cal.save("colqwen2_calibration.npz")
            cal2 = EigenbasisCalibration.load("colqwen2_calibration.npz")
        """
        np.savez_compressed(
            path,
            eigenvectors=self.eigenvectors,
            eigenvalues=self.eigenvalues,
            participation_ratio=np.float64(self.participation_ratio),
            effective_rank=np.int64(self.effective_rank),
            n_samples=np.int64(self.n_samples),
        )

    @classmethod
    def load(cls, path: str | Path) -> EigenbasisCalibration:
        """Load a calibration saved with :meth:`save`."""
        data = np.load(path)
        return cls(
            eigenvectors=data["eigenvectors"].astype(np.float32),
            eigenvalues=data["eigenvalues"].astype(np.float32),
            participation_ratio=float(data["participation_ratio"]),
            effective_rank=int(data["effective_rank"]),
            n_samples=int(data["n_samples"]),
        )


def calibrate_from_embeddings(
    embeddings: np.ndarray,
    center: bool = True,
) -> EigenbasisCalibration:
    """Compute the eigenbasis of the empirical covariance of *embeddings*.

    Parameters
    ----------
    embeddings:
        (N, d) float array of token embeddings from a representative
        calibration corpus.  For ColQwen2, pass the concatenated patch
        embeddings from a sample of documents (e.g. 16–256 pages).
        N ≥ d is required for the covariance to be full rank, but partial
        rank is handled gracefully — eigenvalues are clipped to zero.
    center:
        If True (default), subtract the column mean before computing the
        covariance.  This matches the standard PCA convention and should be
        left on for most use cases.

    Returns
    -------
    EigenbasisCalibration
        Calibration object ready to pass to ``MUVERAEncoder``.

    Raises
    ------
    ValueError
        If *embeddings* is not 2-D or has fewer than 2 rows.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, d), got shape {arr.shape}")
    n, d = arr.shape
    if n < 2:
        raise ValueError(f"At least 2 embedding rows are required for calibration, got {n}")

    if center:
        arr = arr - arr.mean(axis=0, keepdims=True)

    # Covariance: (d, d).  Use float64 internally for numerical stability.
    cov = (arr.astype(np.float64).T @ arr.astype(np.float64)) / max(n - 1, 1)

    # Symmetric eigendecomposition — eigenvalues in ascending order.
    eigenvalues_asc, eigenvectors_asc = np.linalg.eigh(cov)

    # Reverse to descending order.
    eigenvalues = eigenvalues_asc[::-1].astype(np.float32)
    eigenvectors = eigenvectors_asc[:, ::-1].astype(np.float32)

    # Clip numerical negatives from floating-point rounding.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Participation ratio: (Σ λ)² / Σ λ².
    sum_lam = float(eigenvalues.sum())
    sum_lam_sq = float((eigenvalues**2).sum())
    if sum_lam_sq > 0.0:
        participation_ratio = (sum_lam**2) / sum_lam_sq
    else:
        participation_ratio = 1.0

    effective_rank = max(1, round(participation_ratio))

    return EigenbasisCalibration(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        participation_ratio=participation_ratio,
        effective_rank=effective_rank,
        n_samples=n,
    )
