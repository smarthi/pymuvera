"""Tests for the CALIBRATED_EIGENBASIS projection mode.

Covers calibration correctness, encoder integration, ablation toggles,
serialization, and compatibility with existing encoder options.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pymuvera import (
    EigenbasisCalibration,
    MUVERAEncoder,
    ProjectionType,
    calibrate_from_embeddings,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

D = 128
N_CAL = 512  # calibration samples
N_TOK = 64  # tokens per query / document
SEED = 42


def _make_isotropic(n: int, d: int, seed: int = SEED) -> np.ndarray:
    """Isotropic Gaussian embeddings — deff ≈ d."""
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _make_nonisotropic(n: int, d: int, effective_rank: int = 4, seed: int = SEED) -> np.ndarray:
    """Strongly non-isotropic embeddings: variance concentrated in top-r dims.

    Mimics the low-effective-rank structure observed in ColQwen2 patch embeddings
    and reported for LLM key vectors (SpectralQuant, Vangara & Gopinath 2026).
    """
    rng = np.random.default_rng(seed)
    # Build covariance with decaying eigenvalues: lambda_i = 1 / (i+1)^2
    eigenvalues = np.array([1.0 / (i + 1) ** 2 for i in range(d)], dtype=np.float64)
    # Random orthonormal basis
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    cov = (U * eigenvalues[np.newaxis, :]) @ U.T
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    return (rng.standard_normal((n, d)) @ L.T).astype(np.float32)


@pytest.fixture
def cal_embeddings() -> np.ndarray:
    return _make_nonisotropic(N_CAL, D)


@pytest.fixture
def calibration(cal_embeddings: np.ndarray) -> EigenbasisCalibration:
    return calibrate_from_embeddings(cal_embeddings)


@pytest.fixture
def calibrated_encoder(calibration: EigenbasisCalibration) -> MUVERAEncoder:
    enc = MUVERAEncoder(
        dimension=D,
        num_simhash_projections=6,
        num_repetitions=4,
        seed=SEED,
        projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        fill_empty_partitions=True,
    )
    enc.calibrate(_make_nonisotropic(N_CAL, D))
    return enc


# ── calibrate_from_embeddings ─────────────────────────────────────────────


class TestCalibrateFromEmbeddings:
    def test_output_shapes(self, cal_embeddings: np.ndarray) -> None:
        cal = calibrate_from_embeddings(cal_embeddings)
        assert cal.eigenvectors.shape == (D, D)
        assert cal.eigenvalues.shape == (D,)

    def test_eigenvalues_descending(self, calibration: EigenbasisCalibration) -> None:
        assert np.all(calibration.eigenvalues[:-1] >= calibration.eigenvalues[1:])

    def test_eigenvalues_nonnegative(self, calibration: EigenbasisCalibration) -> None:
        assert np.all(calibration.eigenvalues >= 0.0)

    def test_eigenvectors_orthonormal(self, calibration: EigenbasisCalibration) -> None:
        U = calibration.eigenvectors.astype(np.float64)
        err = np.max(np.abs(U.T @ U - np.eye(D)))
        assert err < 1e-4, f"U^T U - I max error: {err}"

    def test_participation_ratio_nonisotropic(self, calibration: EigenbasisCalibration) -> None:
        # Non-isotropic data: PR should be well below d
        assert 1.0 <= calibration.participation_ratio <= D / 2

    def test_participation_ratio_isotropic(self) -> None:
        iso = _make_isotropic(N_CAL, D)
        cal = calibrate_from_embeddings(iso)
        # Isotropic data: PR should be close to d
        assert cal.participation_ratio > D * 0.5

    def test_n_samples_recorded(self, cal_embeddings: np.ndarray) -> None:
        cal = calibrate_from_embeddings(cal_embeddings)
        assert cal.n_samples == len(cal_embeddings)

    def test_effective_rank_positive(self, calibration: EigenbasisCalibration) -> None:
        assert calibration.effective_rank >= 1

    def test_raises_on_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            calibrate_from_embeddings(np.ones(D, dtype=np.float32))

    def test_raises_on_single_row(self) -> None:
        with pytest.raises(ValueError, match="2"):
            calibrate_from_embeddings(np.ones((1, D), dtype=np.float32))

    def test_no_centering(self, cal_embeddings: np.ndarray) -> None:
        cal = calibrate_from_embeddings(cal_embeddings, center=False)
        assert cal.eigenvectors.shape == (D, D)


# ── EigenbasisCalibration serialization ───────────────────────────────────


class TestCalibrationSerialization:
    def test_save_load_round_trip(self, calibration: EigenbasisCalibration) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.npz"
            calibration.save(path)
            loaded = EigenbasisCalibration.load(path)

        np.testing.assert_allclose(loaded.eigenvectors, calibration.eigenvectors, atol=1e-6)
        np.testing.assert_allclose(loaded.eigenvalues, calibration.eigenvalues, atol=1e-6)
        assert loaded.participation_ratio == pytest.approx(
            calibration.participation_ratio, rel=1e-6
        )
        assert loaded.effective_rank == calibration.effective_rank
        assert loaded.n_samples == calibration.n_samples

    def test_save_produces_npz(self, calibration: EigenbasisCalibration) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.npz"
            calibration.save(path)
            assert path.exists()


# ── MUVERAEncoder.calibrate() ─────────────────────────────────────────────


class TestEncoderCalibrate:
    def test_calibrate_sets_calibration(self, cal_embeddings: np.ndarray) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        )
        assert enc.calibration is None
        enc.calibrate(cal_embeddings)
        assert enc.calibration is not None

    def test_calibrate_returns_self(self, cal_embeddings: np.ndarray) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        )
        result = enc.calibrate(cal_embeddings)
        assert result is enc

    def test_calibrate_builds_rep_params(self, cal_embeddings: np.ndarray) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=3,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        )
        enc.calibrate(cal_embeddings)
        assert len(enc._rep_params) == 3

    def test_calibrate_raises_on_wrong_projection_type(self) -> None:
        enc = MUVERAEncoder(dimension=D, num_simhash_projections=4)
        with pytest.raises(ValueError, match="CALIBRATED_EIGENBASIS"):
            enc.calibrate(np.ones((100, D), dtype=np.float32))

    def test_calibration_passed_at_construction(self, calibration: EigenbasisCalibration) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
            calibration=calibration,
        )
        assert enc.calibration is not None
        assert len(enc._rep_params) == 2

    def test_uncalibrated_encode_query_raises(self) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        )
        with pytest.raises(RuntimeError, match="calibrate"):
            enc.encode_query(np.ones((N_TOK, D), dtype=np.float32))

    def test_uncalibrated_encode_document_raises(self) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        )
        with pytest.raises(RuntimeError, match="calibrate"):
            enc.encode_document(np.ones((N_TOK, D), dtype=np.float32))


# ── Output shape and basic correctness ────────────────────────────────────


class TestCalibratedEigenbasisEncoding:
    def test_query_fde_shape(self, calibrated_encoder: MUVERAEncoder) -> None:
        tokens = _make_nonisotropic(N_TOK, D, seed=1)
        q_fde = calibrated_encoder.encode_query(tokens)
        assert q_fde.shape == (calibrated_encoder.fde_dimension,)
        assert q_fde.dtype == np.float32

    def test_document_fde_shape(self, calibrated_encoder: MUVERAEncoder) -> None:
        tokens = _make_nonisotropic(N_TOK, D, seed=2)
        d_fde = calibrated_encoder.encode_document(tokens)
        assert d_fde.shape == (calibrated_encoder.fde_dimension,)
        assert d_fde.dtype == np.float32

    def test_fde_dimension_formula(self) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=5,
            num_repetitions=3,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
            calibration=calibrate_from_embeddings(_make_nonisotropic(N_CAL, D)),
        )
        expected = 3 * (2**5) * D
        assert enc.fde_dimension == expected

    def test_dot_product_positive_for_similar_inputs(
        self, calibrated_encoder: MUVERAEncoder
    ) -> None:
        """FDE dot product should be positive when query and document are similar."""
        tokens = _make_nonisotropic(N_TOK, D, seed=10)
        # Perturb slightly for document
        doc_tokens = tokens + 0.01 * np.random.default_rng(99).standard_normal(tokens.shape).astype(
            np.float32
        )
        q_fde = calibrated_encoder.encode_query(tokens)
        d_fde = calibrated_encoder.encode_document(doc_tokens)
        assert float(q_fde @ d_fde) > 0.0

    def test_deterministic_encoding(self, calibrated_encoder: MUVERAEncoder) -> None:
        tokens = _make_nonisotropic(N_TOK, D, seed=7)
        fde1 = calibrated_encoder.encode_query(tokens)
        fde2 = calibrated_encoder.encode_query(tokens)
        np.testing.assert_array_equal(fde1, fde2)

    def test_batch_encode_queries(self, calibrated_encoder: MUVERAEncoder) -> None:
        batch = [_make_nonisotropic(N_TOK, D, seed=i) for i in range(5)]
        Q = calibrated_encoder.encode_queries_batch(batch)
        assert Q.shape == (5, calibrated_encoder.fde_dimension)

    def test_batch_encode_documents(self, calibrated_encoder: MUVERAEncoder) -> None:
        batch = [_make_nonisotropic(N_TOK, D, seed=i + 10) for i in range(5)]
        D_mat = calibrated_encoder.encode_documents_batch(batch)
        assert D_mat.shape == (5, calibrated_encoder.fde_dimension)


# ── Ablation: eigenvalue weighting toggle ─────────────────────────────────


class TestEigenvalueWeightingToggle:
    def _make_encoder(self, use_eigenvalue_weighting: bool) -> MUVERAEncoder:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=6,
            num_repetitions=4,
            seed=SEED,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
            use_eigenvalue_weighting=use_eigenvalue_weighting,
        )
        enc.calibrate(_make_nonisotropic(N_CAL, D))
        return enc

    def test_weighted_and_unweighted_produce_different_fdEs(self) -> None:
        tokens = _make_nonisotropic(N_TOK, D, seed=42)
        enc_w = self._make_encoder(use_eigenvalue_weighting=True)
        enc_u = self._make_encoder(use_eigenvalue_weighting=False)
        fde_w = enc_w.encode_query(tokens)
        fde_u = enc_u.encode_query(tokens)
        assert not np.allclose(fde_w, fde_u), (
            "Eigenvalue-weighted and uniform-projection FDEs should differ"
        )

    def test_both_produce_correct_shape(self) -> None:
        tokens = _make_nonisotropic(N_TOK, D, seed=5)
        for ew in (True, False):
            enc = self._make_encoder(ew)
            fde = enc.encode_query(tokens)
            assert fde.shape == (enc.fde_dimension,)

    def test_eigenvalue_weighting_scales_simhash_rows_by_sqrt_lambda(self) -> None:
        """With eigenvalue weighting, the SimHash matrix rows in eigenbasis space
        should have L2 norms proportional to sqrt(lambda_i).

        This directly tests that the water-filling analog is implemented correctly:
        row i of the (d, k) projection matrix has expected squared norm
        k * lambda_i, so its expected L2 norm ~ sqrt(k * lambda_i).
        """
        cal_embs = _make_nonisotropic(N_CAL, D)
        calibration = calibrate_from_embeddings(cal_embs)

        # Build a single rep_params with eigenvalue weighting on
        from pymuvera._internal.params import build_rep_params

        params = build_rep_params(
            rep_seed=SEED,
            dimension=D,
            projection_dim=D,
            num_simhash_projections=32,  # many projections for stable row norms
            use_identity=True,
            use_calibrated_eigenbasis=True,
            calibration_eigenvalues=calibration.eigenvalues,
            calibration_eigenvectors=calibration.eigenvectors,
            use_eigenvalue_weighting=True,
        )
        W = params.eigenbasis_simhash_mat  # (d, k)
        assert W is not None

        # Row norms should correlate with sqrt(lambda_i)
        row_norms = np.linalg.norm(W, axis=1)  # (d,)
        expected_scale = np.sqrt(calibration.eigenvalues)  # (d,)

        # Mask out near-zero eigenvalues (tail dimensions)
        mask = calibration.eigenvalues > calibration.eigenvalues[0] * 1e-3
        correlation = float(np.corrcoef(row_norms[mask], expected_scale[mask])[0, 1])
        assert correlation > 0.9, (
            f"Row norms of W should correlate with sqrt(lambda), got r={correlation:.3f}"
        )

    def test_unweighted_simhash_rows_uniform(self) -> None:
        """Without eigenvalue weighting, row norms of the SimHash matrix should
        be approximately uniform (standard Gaussian, scale 1).
        """
        cal_embs = _make_nonisotropic(N_CAL, D)
        calibration = calibrate_from_embeddings(cal_embs)

        from pymuvera._internal.params import build_rep_params

        params = build_rep_params(
            rep_seed=SEED,
            dimension=D,
            projection_dim=D,
            num_simhash_projections=32,
            use_identity=True,
            use_calibrated_eigenbasis=True,
            calibration_eigenvalues=calibration.eigenvalues,
            calibration_eigenvectors=calibration.eigenvectors,
            use_eigenvalue_weighting=False,
        )
        W = params.eigenbasis_simhash_mat
        assert W is not None

        row_norms = np.linalg.norm(W, axis=1)
        expected_scale = np.sqrt(calibration.eigenvalues)

        # Row norms should NOT correlate with sqrt(lambda) — uniform sampling
        mask = calibration.eigenvalues > calibration.eigenvalues[0] * 1e-3
        correlation = float(np.corrcoef(row_norms[mask], expected_scale[mask])[0, 1])
        # Weak correlation only — uniform projection has no systematic eigenvalue dependence
        assert abs(correlation) < 0.5, (
            f"Unweighted SimHash rows should not correlate with eigenvalues, got r={correlation:.3f}"
        )


# ── Compatibility with existing encoder options ───────────────────────────


class TestCompatibilityWithExistingOptions:
    def _calibrated_enc(self, **kwargs) -> MUVERAEncoder:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=6,
            num_repetitions=2,
            seed=SEED,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
            **kwargs,
        )
        enc.calibrate(_make_nonisotropic(N_CAL, D))
        return enc

    def test_with_fill_empty_partitions(self) -> None:
        enc = self._calibrated_enc(fill_empty_partitions=True)
        d_fde = enc.encode_document(np.ones((3, D), dtype=np.float32))
        assert d_fde.shape == (enc.fde_dimension,)
        assert not np.any(np.isnan(d_fde))

    def test_with_densifying_fill(self) -> None:
        enc = self._calibrated_enc(fill_empty_partitions=True, densifying_fill=True)
        d_fde = enc.encode_document(np.ones((3, D), dtype=np.float32))
        assert d_fde.shape == (enc.fde_dimension,)

    def test_with_final_projection_dimension(self) -> None:
        target_dim = 512
        enc = self._calibrated_enc(final_projection_dimension=target_dim)
        assert enc.fde_dimension == target_dim
        q_fde = enc.encode_query(_make_nonisotropic(N_TOK, D))
        d_fde = enc.encode_document(_make_nonisotropic(N_TOK, D, seed=2))
        assert q_fde.shape == (target_dim,)
        assert d_fde.shape == (target_dim,)

    def test_repr_shows_calibration_info(self) -> None:
        enc = self._calibrated_enc()
        r = repr(enc)
        assert "CALIBRATED_EIGENBASIS" in r
        assert "calibrated=True" in r
        assert "participation_ratio=" in r

    def test_repr_uncalibrated_shows_not_calibrated(self) -> None:
        enc = MUVERAEncoder(
            dimension=D,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
        )
        r = repr(enc)
        assert "calibrated=False" in r
