"""
Public configuration types for muvera-fde.

These are the only types callers need to import to configure an encoder.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict


class ProjectionType(enum.Enum):
    """Projection strategy applied to token embeddings before SimHash partitioning.

    Naming note
    -----------
    ``AMS_SKETCH`` is a misnomer inherited from the Google graph-mining C++ source.
    The actual construction is a **Count Sketch** (Charikar, Chen & Farach-Colton, 2002),
    not an AMS sketch (Alon-Matias-Szegedy, 1996).
    """

    DEFAULT_IDENTITY = 0
    """Full-rank Gaussian SimHash baseline.

    Cost: O(N x d x k) per repetition.
    Constraint: none.
    """

    AMS_SKETCH = 1
    """Count Sketch token projection before full-rank Gaussian SimHash.

    Cost: O(N x d) projection + O(N x projection_dim x k) SimHash.
    Constraint: projection_dimension required.
    """

    LOW_RANK_GAUSSIAN = 2
    """Low-rank Gaussian SimHash: W ~ AB^T, A in R^{d x r}, B in R^{k x r}.

    Reduces cost from O(N x d x k) to O(N x d x r + N x r x k).
    Sign-pattern agreement with full-rank improves as r/k decreases.
    Use r/k <= 0.25 as a practical guideline. Formal bounds are an open problem.

    Cost: O(N x r x (d + k)).
    Constraint: 1 <= simhash_rank < num_simhash_projections.
    """

    SRHT = 3
    """Subsampled Randomized Hadamard Transform.

    Applies S H D x: Rademacher sign flip, Walsh-Hadamard butterfly,
    random row subsampling. The linear projection step satisfies the
    JL lemma (Woolfe et al. 2008; Tropp 2011 arXiv:1011.1595).
    Note: SimHash applies sign() which is nonlinear; the JL result
    governs the projection quality, not partition assignments directly.

    Cost: O(N x d' x log(d')) independent of k.
    Constraint: num_simhash_projections <= next_power_of_2(dimension).
    """

    CROSS_POLYTOPE = 4
    """Cross-Polytope LSH: argmax(|H D x|) partition assignment.

    Applies a full SRHT rotation (no subsampling) then assigns each token
    to its dominant coordinate. Theoretically optimal for cosine similarity
    (Andoni & Razenshteyn, 2015).

    num_partitions = 2 * next_power_of_2(dimension) per repetition.
    num_simhash_projections is IGNORED.
    Densifying fill is used automatically (no sketch matrix for Hamming).

    Cost: O(N x d' x log(d')).
    """

    CALIBRATED_EIGENBASIS = 5
    """Data-aware eigenbasis rotation with optional eigenvalue-weighted SimHash.

    Rotates embeddings into the eigenbasis of the empirical token covariance
    before SimHash partitioning.  The stored FDE slot centroids live in the
    eigenbasis space; inner products are preserved exactly since the rotation
    is orthogonal.

    With ``use_eigenvalue_weighting=True`` (default), the SimHash projection
    matrix is sampled from N(0, diag(lambda)) in the rotated space, concentrating
    partition discrimination on the high-variance eigenbasis coordinates — the
    SimHash analog of the water-filling bit allocation in SpectralQuant
    (Vangara & Gopinath, 2026).

    Requires calibration: call ``MUVERAEncoder.calibrate(embeddings)`` or pass
    an ``EigenbasisCalibration`` at construction time before encoding.

    Cost: O(N x d^2) for rotation + O(N x d x k) for SimHash.
    Constraint: ``num_simhash_projections >= 1``.
    """


class FDEConfig(BaseModel):
    """Immutable configuration for Fixed Dimensional Encoding."""

    model_config = ConfigDict(frozen=True)

    dimension: int = 128
    num_repetitions: int = 1
    num_simhash_projections: int = 4
    seed: int = 1
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: int | None = None
    simhash_rank: int = 1
    fill_empty_partitions: bool = False
    densifying_fill: bool = False
    final_projection_dimension: int | None = None
    use_eigenvalue_weighting: bool = True
    """Only active when projection_type == CALIBRATED_EIGENBASIS.

    If True (default), SimHash rows are scaled by sqrt(lambda_i) in the
    eigenbasis so that high-variance coordinates dominate bucket assignment.
    If False, uniform Gaussian SimHash is applied in the eigenbasis
    (ablation baseline isolating the rotation contribution from the
    eigenvalue-weighted projection contribution).
    """
