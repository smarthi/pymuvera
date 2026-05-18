"""High-level encoder for muvera-fde."""

from __future__ import annotations

import numpy as np

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
    """Encodes multi-vector token embeddings into fixed-dimensional FDE vectors."""

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
        )
        self.fill_empty_partitions = fill_empty_partitions

        config = FDEConfig(**self._base_config, fill_empty_partitions=fill_empty_partitions)
        validate_config(config)

        _proj_dim = _projection_dim_for(config)

        self._rep_params: list[RepParams] = [
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

    @property
    def fde_dimension(self) -> int:
        if self._base_config["final_projection_dimension"] is not None:
            return self._base_config["final_projection_dimension"]
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        proj_dim = _projection_dim_for(config)
        return config.num_repetitions * num_partitions_for_config(config, proj_dim) * proj_dim

    def encode_query(self, token_embeddings: np.ndarray) -> np.ndarray:
        config = FDEConfig(**self._base_config, fill_empty_partitions=False)
        return generate_query_fde(token_embeddings, config, self._rep_params)

    def encode_document(self, token_embeddings: np.ndarray) -> np.ndarray:
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
