"""High-level encoder for muvera-fde."""

from __future__ import annotations

import numpy as np

from pymuvera._internal.calibration import EigenbasisCalibration, calibrate_from_embeddings
from pymuvera._internal.params import RepParams, build_rep_params
from pymuvera._internal.validation import (
    num_partitions_for_config,
    validate_config,
)
from pymuvera.config import FDEConfig, ProjectionType
from pymuvera.core import (
    _projection_dim_for,
    _use_cross_polytope,
    _use_identity,
    _use_low_rank_simhash,
    _use_srht,
    generate_document_fde,
    generate_query_fde,
)


class MUVERAEncoder:
    """Encodes multi-vector token embeddings into fixed-dimensional FDE vectors.

    For ``CALIBRATED_EIGENBASIS`` mode, either pass an ``EigenbasisCalibration``
    at construction time or call :meth:`calibrate` with a representative sample
    of embeddings before encoding::

        enc = MUVERAEncoder(
            dimension=128,
            num_simhash_projections=8,
            num_repetitions=8,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
            fill_empty_partitions=True,
        )
        enc.calibrate(calibration_embeddings)   # shape: (N, 128)
        q_fde = enc.encode_query(query_tokens)
        d_fde = enc.encode_document(doc_tokens)
    """

    def __init__(
        self,
        dimension: int = 128,
        num_simhash_projections: int = 4,
        num_repetitions: int = 1,
        seed: int = 1,
        projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY,
        projection_dimension: int | None = None,
        simhash_rank: int = 1,
        fill_empty_partitions: bool = False,
        densifying_fill: bool = False,
        final_projection_dimension: int | None = None,
        use_eigenvalue_weighting: bool = True,
        calibration: EigenbasisCalibration | None = None,
    ) -> None:
        self._base_config: dict = dict(
            dimension=dimension,
            num_repetitions=num_repetitions,
            num_simhash_projections=num_simhash_projections,
            seed=seed,
            projection_type=projection_type,
            projection_dimension=projection_dimension,
            simhash_rank=simhash_rank,
            densifying_fill=densifying_fill,
            final_projection_dimension=final_projection_dimension,
            use_eigenvalue_weighting=use_eigenvalue_weighting,
        )
        self.fill_empty_partitions = fill_empty_partitions
        self._calibration: EigenbasisCalibration | None = None

        config = FDEConfig(**self._base_config, fill_empty_partitions=fill_empty_partitions)
        validate_config(config)

        if projection_type == ProjectionType.CALIBRATED_EIGENBASIS:
            if calibration is not None:
                self._calibration = calibration
                self._rep_params = self._build_calibrated_rep_params(config, calibration)
            else:
                # Deferred: _rep_params will be built after calibrate() is called.
                self._rep_params = []
        else:
            _proj_dim = _projection_dim_for(config)
            self._rep_params = [
                build_rep_params(
                    seed + rep,
                    dimension,
                    _proj_dim,
                    num_simhash_projections,
                    _use_identity(config),
                    use_low_rank_simhash=_use_low_rank_simhash(config),
                    simhash_rank=simhash_rank,
                    use_srht=_use_srht(config),
                    use_cross_polytope=_use_cross_polytope(config),
                )
                for rep in range(num_repetitions)
            ]

    # ── Calibration ───────────────────────────────────────────────────────

    def calibrate(
        self,
        embeddings: np.ndarray,
        center: bool = True,
    ) -> MUVERAEncoder:
        """Compute the eigenbasis calibration from *embeddings* and rebuild params.

        Parameters
        ----------
        embeddings:
            (N, d) float array of representative token embeddings.  For ColQwen2,
            pass the concatenated patch embeddings from 16–256 document pages.
            N ≥ d is recommended; N ≥ 256 gives stable eigenvalue estimates for
            d=128.
        center:
            If True (default), subtract the column mean before computing the
            covariance.

        Returns
        -------
        self
            Returns the encoder for method chaining.

        Raises
        ------
        ValueError
            If the encoder is not configured with ``CALIBRATED_EIGENBASIS``.
        """
        if self._base_config["projection_type"] != ProjectionType.CALIBRATED_EIGENBASIS:
            raise ValueError(
                "calibrate() is only supported for CALIBRATED_EIGENBASIS encoders. "
                f"Got projection_type={self._base_config['projection_type'].name}."
            )
        self._calibration = calibrate_from_embeddings(embeddings, center=center)
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        self._rep_params = self._build_calibrated_rep_params(config, self._calibration)
        return self

    def _build_calibrated_rep_params(
        self,
        config: FDEConfig,
        calibration: EigenbasisCalibration,
    ) -> list[RepParams]:
        _proj_dim = _projection_dim_for(config)
        return [
            build_rep_params(
                config.seed + rep,
                config.dimension,
                _proj_dim,
                config.num_simhash_projections,
                use_identity=True,  # no AMS pre-projection for CALIBRATED_EIGENBASIS
                use_calibrated_eigenbasis=True,
                calibration_eigenvalues=calibration.eigenvalues,
                calibration_eigenvectors=calibration.eigenvectors,
                use_eigenvalue_weighting=config.use_eigenvalue_weighting,
            )
            for rep in range(config.num_repetitions)
        ]

    def _assert_calibrated(self) -> None:
        if self._base_config["projection_type"] == ProjectionType.CALIBRATED_EIGENBASIS and (
            not self._rep_params
        ):
            raise RuntimeError(
                "MUVERAEncoder with CALIBRATED_EIGENBASIS has not been calibrated. "
                "Call calibrate(embeddings) before encoding, or pass a pre-computed "
                "EigenbasisCalibration to the constructor."
            )

    # ── Encoding ──────────────────────────────────────────────────────────

    @property
    def calibration(self) -> EigenbasisCalibration | None:
        """The current calibration, or None if not yet calibrated."""
        return self._calibration

    @property
    def fde_dimension(self) -> int:
        if self._base_config["final_projection_dimension"] is not None:
            return self._base_config["final_projection_dimension"]
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        proj_dim = _projection_dim_for(config)
        return config.num_repetitions * num_partitions_for_config(config, proj_dim) * proj_dim

    def encode_query(self, token_embeddings: np.ndarray) -> np.ndarray:
        self._assert_calibrated()
        config = FDEConfig(**self._base_config, fill_empty_partitions=False)
        return generate_query_fde(token_embeddings, config, self._rep_params)

    def encode_document(self, token_embeddings: np.ndarray) -> np.ndarray:
        self._assert_calibrated()
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        return generate_document_fde(token_embeddings, config, self._rep_params)

    def encode_queries_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.encode_query(q) for q in batch])

    def encode_documents_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.encode_document(d) for d in batch])

    def __repr__(self) -> str:
        cfg = self._base_config
        pt = cfg["projection_type"]
        extra = ""
        if pt == ProjectionType.LOW_RANK_GAUSSIAN:
            extra = f", simhash_rank={cfg['simhash_rank']}"
        if pt == ProjectionType.CALIBRATED_EIGENBASIS:
            ew = "on" if cfg.get("use_eigenvalue_weighting", True) else "off"
            calibrated = self._calibration is not None
            pr = (
                f"{self._calibration.participation_ratio:.1f}"
                if self._calibration is not None
                else "?"
            )
            extra = (
                f", eigenvalue_weighting={ew}, calibrated={calibrated}, participation_ratio={pr}"
            )
        if cfg.get("densifying_fill"):
            extra += ", densifying_fill=True"
        return (
            f"MUVERAEncoder("
            f"dimension={cfg['dimension']}, "
            f"num_simhash_projections={cfg['num_simhash_projections']}, "
            f"num_repetitions={cfg['num_repetitions']}, "
            f"projection_type={pt.name}"
            f"{extra}, "
            f"fde_dimension={self.fde_dimension}"
            f")"
        )
