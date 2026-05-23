# pymuvera — MUVERA: Fixed Dimensional Encodings for Multi-Vector Retrieval

**Sublinear ANN retrieval for ColBERT, ColPali, ColQwen2, and ColQwen3.5.**

[![PyPI](https://img.shields.io/pypi/v/pymuvera)](https://pypi.org/project/pymuvera/)
[![Python](https://img.shields.io/pypi/pyversions/pymuvera)](https://pypi.org/project/pymuvera/)
[![CI](https://github.com/smarthi/muvera-fde/actions/workflows/ci.yml/badge.svg)](https://github.com/smarthi/muvera-fde/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A pure-Python port of Google's graph-mining MUVERA implementation, extended with
**low-rank SimHash factorisation** (inspired by Sarkar et al., 2025),
**Subsampled Randomized Hadamard Transform** (SRHT, Woolfe, Liberty, Rokhlin & Tygert, 2008),
**Cross-Polytope LSH** (Andoni & Razenshteyn, 2015),
**Densifying LSH fill** (Shrivastava, 2014), and
**Calibrated Eigenbasis SimHash** with eigenvalue-weighted partitioning (inspired by SpectralQuant, Vangara & Gopinath, 2026).

| | Reference |
|---|---|
| MUVERA paper | [Dhulipala et al., 2024](https://arxiv.org/abs/2405.19504) |
| LOW_RANK_GAUSSIAN inspiration | [Sarkar et al., 2025](https://eshyperscale.github.io/imgs/paper.pdf) |
| SRHT | [Woolfe, Liberty, Rokhlin & Tygert, 2008](https://doi.org/10.1016/j.acha.2007.12.002) |
| Cross-Polytope LSH | [Andoni & Razenshteyn, 2015](https://arxiv.org/abs/1509.02897) |
| Densifying LSH | [Shrivastava, 2014](https://arxiv.org/abs/1401.4605) |
| CALIBRATED_EIGENBASIS inspiration | [Vangara & Gopinath, 2026](https://arxiv.org/abs/2506.nnnnn) |
| Original C++ implementation | [google/graph-mining](https://github.com/google/graph-mining/tree/main/sketching/point_cloud) |

---

## What this library adds beyond the original paper

The MUVERA paper uses a full-rank Gaussian matrix for SimHash partitioning and
Hamming nearest-neighbor fill for empty partitions. This library adds four new
capabilities:

**`LOW_RANK_GAUSSIAN`** factors the SimHash matrix as AB⊤ (`A ∈ ℝ^{d×r}`,
`B ∈ ℝ^{k×r}`, `r ≪ k`), cutting partition cost from `O(N·d·k)` to
`O(N·d·r + N·r·k)`. Inspired by Sarkar et al. (2025); formal partition quality
bounds are an open problem. At r=4, ColQwen2 (d=128, k=8): **~1.9× faster**.

**`SRHT`** (Woolfe et al., 2008) applies a structured `S·H·D` transform at
`O(N·d·log d)` cost, independent of k. **Full JL guarantee**, zero rank error.
For ColQwen2 (d=128, k=8): 904N vs 1024N ops.

**`CROSS_POLYTOPE`** (Andoni & Razenshteyn, 2015) uses `argmax(|H·D·x|)` instead
of sign-based SimHash, producing 2·padded_dim partitions per repetition aligned with
the Voronoi cells of the cross-polytope — **theoretically optimal for cosine
similarity** in high dimensions. For ColQwen2 (d=128): 256 partitions at O(d log d)
cost. For ColQwen3.5 (d=320): 1024 partitions.

**Densifying LSH fill** (Shrivastava, 2014) replaces O(N·k) Hamming nearest-neighbor
fill with a deterministic O(num_empty) hash-based fill. No sketch matrix needed —
automatically used for `CROSS_POLYTOPE`, opt-in for other modes via `densifying_fill=True`.

**`CALIBRATED_EIGENBASIS`** rotates embeddings into the eigenbasis of the empirical
token covariance before SimHash partitioning, concentrating partition discrimination
on the few high-variance directions that trained attention heads actually use.  With
`use_eigenvalue_weighting=True` (default), the SimHash projection matrix is sampled
from N(0, diag(λ)) in the rotated space — the water-filling analog from rate–distortion
theory applied to FDE partitioning.  Inspired by SpectralQuant's finding (Vangara &
Gopinath, 2026) that LLM key covariances have participation ratio deff/dh ≈ 3–5% at
dh=128, a structure expected to hold in ColQwen2/ColQwen3.5 patch embeddings.
Requires a one-time `calibrate(embeddings)` call; calibration objects are serializable
alongside your ANN index.

---

## What is MUVERA?

Late-interaction retrieval models like **ColBERT**, **ColPali**, and **ColQwen2**
represent each query and document as a *variable-length set* of token embeddings
rather than a single vector. Scoring two sets requires the computationally
expensive **MaxSim** (Chamfer Similarity) operation:

```
Chamfer(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} cos(q, d)
```

This makes large-scale ANN retrieval impractical with standard indexes.

MUVERA solves this by converting each multi-vector set into a **single
fixed-dimensional vector** (FDE) such that:

```
fde_query(Q) · fde_doc(D)  ≈  Chamfer(Q, D)
```

Standard ANN libraries (FAISS, ScaNN, OpenSearch k-NN) can then index FDE
vectors directly, restoring sublinear retrieval for late-interaction models.

---

## Installation

```bash
pip install pymuvera
```

Requires Python ≥ 3.12, NumPy ≥ 1.24, Pydantic ≥ 2.0.

---

## Quick start

```python
import numpy as np
from pymuvera import MUVERAEncoder

# One encoder instance for both queries and documents — seed must match
enc = MUVERAEncoder(
  dimension=128,  # ColBERT / ColQwen2 token embedding dimension
  num_simhash_projections=4,  # 2^4 = 16 partitions per repetition
  num_repetitions=2,  # 2 independent repetitions
  seed=42,
)

print(enc)
# MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2,
#               projection_type=DEFAULT_IDENTITY, fde_dimension=4096)

query_tokens = np.random.randn(32, 128).astype(np.float32)  # 32 query tokens
doc_tokens = np.random.randn(512, 128).astype(np.float32)  # 512 document tokens

q_fde = enc.encode_query(query_tokens)  # shape: (4096,)
d_fde = enc.encode_document(doc_tokens)  # shape: (4096,)

# Approximate Chamfer Similarity — drop into any ANN index as a float32 vector
score = float(q_fde @ d_fde)
```

---

## API reference

### `MUVERAEncoder`

The primary entry point. Initialize **once** and reuse for all queries and
documents — the random partition structure (SimHash matrices, Count Sketch
parameters) must be identical on both sides.

```python
MUVERAEncoder(
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
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimension` | 128 | Token embedding dimension |
| `num_simhash_projections` | 4 | SimHash bits *k*; partitions = 2^k |
| `num_repetitions` | 1 | Independent repetitions (more → better approximation) |
| `seed` | 1 | Shared RNG seed — **must match** query and document sides |
| `projection_type` | `DEFAULT_IDENTITY` | `DEFAULT_IDENTITY`, `AMS_SKETCH`, `LOW_RANK_GAUSSIAN`, `SRHT`, `CROSS_POLYTOPE`, or `CALIBRATED_EIGENBASIS` |
| `projection_dimension` | `None` | Target dim after Count Sketch; required for `AMS_SKETCH` |
| `simhash_rank` | 1 | Rank *r* for `LOW_RANK_GAUSSIAN`; must satisfy `1 ≤ r < num_simhash_projections` |
| `fill_empty_partitions` | `False` | Document side: fill empty slots |
| `densifying_fill` | `False` | Use O(num_empty) Densifying LSH fill. Automatically forced True for `CROSS_POLYTOPE` |
| `final_projection_dimension` | `None` | Post-accumulation Count Sketch compression |
| `use_eigenvalue_weighting` | `True` | `CALIBRATED_EIGENBASIS` only: scale SimHash rows by √λ_i so high-variance eigendirections dominate bucket assignment (water-filling analog). Set False for ablation |
| `calibration` | `None` | `CALIBRATED_EIGENBASIS` only: pre-computed `EigenbasisCalibration`. Alternative to calling `calibrate()` post-construction |

**Property:** `fde_dimension` — output vector length.

---

### Encoding single inputs

```python
enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2)

# Query: SUM aggregation — token embeddings summed into their SimHash partition
q_fde = enc.encode_query(query_tokens)    # (num_tokens, 128) → (fde_dim,)

# Document: AVERAGE aggregation — centroid of tokens per partition
d_fde = enc.encode_document(doc_tokens)   # (num_tokens, 128) → (fde_dim,)

# Both also accept flat 1-D input (num_tokens * dimension,)
q_fde = enc.encode_query(query_tokens.flatten())
```

---

### Batch encoding

```python
queries   = [np.random.randn(32,  128).astype(np.float32) for _ in range(100)]
documents = [np.random.randn(512, 128).astype(np.float32) for _ in range(1000)]

Q = enc.encode_queries_batch(queries)     # shape: (100,  fde_dimension)
D = enc.encode_documents_batch(documents) # shape: (1000, fde_dimension)

# All-pairs approximate Chamfer Similarities in one matmul
scores = Q @ D.T   # shape: (100, 1000)
top_k  = np.argsort(scores, axis=1)[:, ::-1][:, :10]  # top-10 per query
```

---

### Reducing FDE size

Two orthogonal compression knobs:

**Option A — per-partition Count Sketch** (reduces width before accumulation):

```python
from pymuvera import ProjectionType

enc = MUVERAEncoder(
  dimension=128,
  num_simhash_projections=4,
  num_repetitions=4,
  projection_type=ProjectionType.AMS_SKETCH,
  projection_dimension=32,  # 128 → 32 per partition slot
)
# fde_dimension = 4 reps × 16 partitions × 32 = 2048  (vs 8192 without)
```

**Option B — post-accumulation Count Sketch** (compresses the final vector):

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=4,
    final_projection_dimension=512,   # 8192 → 512
)
# fde_dimension = 512
```

Both preserve dot products in expectation: `E[⟨sketch(x), sketch(y)⟩] = ⟨x, y⟩`.

---

### SimHash projection modes

Three SimHash projection modes are available, each trading speed against quality.
All produce the **same FDE output shape** and are **drop-in replacements** for
each other — only the SimHash matrix computation changes.

#### Mode 1: `DEFAULT_IDENTITY` — full-rank Gaussian (baseline)

Samples a fresh `(d × k)` Gaussian matrix per repetition. JL guarantee,
full-rank quality. Baseline for comparison.

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=4,
)
# SimHash cost: O(N × 128 × 8) = 1024N ops/rep
```

---

#### Mode 2: `LOW_RANK_GAUSSIAN` — low-rank factored SimHash

Factors `W ≈ AB⊤` where `A ∈ ℝ^{d×r}`, `B ∈ ℝ^{k×r}`, replacing one large
matmul with two smaller ones:

```python
from pymuvera import ProjectionType

enc = MUVERAEncoder(
  dimension=128,
  num_simhash_projections=8,
  num_repetitions=4,
  projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
  simhash_rank=4,  # r=4: O(N×128×4 + N×4×8) = 544N ops — 1.9× faster
  seed=42,
)
```

**Computational motivation:** the low-rank factorisation reduces SimHash cost
from O(N·d·k) to O(N·d·r + N·r·k). Sign-pattern agreement with full-rank
Gaussian SimHash improves as r/k decreases. Inspired by Sarkar et al. (2025),
though their theoretical guarantees for evolution strategy updates do not
transfer directly to the SimHash partition setting. Formal partition quality
bounds are an open problem; use **r/k ≤ 0.25** as a practical guideline.

| `simhash_rank` r | CLT baseline O(r⁻¹/²) | LOW_RANK_GAUSSIAN O(r⁻¹) (empirical) | Speedup vs baseline |
|---|---|---|---|
| 4 | ~50% error | **~25% error** | 1.9× |
| 9 | ~33% error | **~11% error** | — |
| 16 | ~25% error | **~6% error** | — |

Cost breakdown for ColQwen2 (d=128, k=8):

| `simhash_rank` | SimHash cost | Speedup |
|---|---|---|
| 1 | 136N ops | 7.5× |
| 4 | 544N ops | 1.9× |
| 8 | 1088N ops | ~breakeven |

> The 1/√r normalisation is omitted — SimHash sign assignments are
> scale-invariant (`sign(αx) = sign(x)`), so it has no effect.

---

#### Mode 3: `SRHT` — Subsampled Randomized Hadamard Transform

Applies the structured transform `S·H·D` row-wise:

* **D** — random diagonal ±1 (Rademacher sign flip)
* **H** — Walsh-Hadamard transform (O(d log d) butterfly)
* **S** — random row subsampling to k dimensions

Input is zero-padded to the next power of 2 ≥ d before applying H.

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=4,
    projection_type=ProjectionType.SRHT,
    seed=42,
)
# SimHash cost: O(N × 128 × log₂(128) + N × 8) = O(N × 128 × 7 + N × 8) = 904N ops
# No rank approximation error — full JL guarantee (Woolfe, Liberty, Rokhlin & Tygert, 2008)
# Constraint: num_simhash_projections <= next_power_of_2(dimension)
```

**Theoretical guarantee**: SRHT is a full Johnson-Lindenstrauss projection —
it preserves pairwise distances to ε with high probability, with no rank
approximation error. Unlike LOW_RANK_GAUSSIAN, it converges exactly to
full-rank Gaussian quality at `k = d`.
Tropp (2011) provides the tightest known analysis, proving that
`ℓ ≥ (1+ι) · k log(k)` subsampled dimensions suffice to preserve an entire
k-dimensional subspace with optimal constants via matrix Chernoff inequalities.
For SimHash (sign-only) use, this subspace result is sufficient but not tight —
sign assignments are scale-invariant so the embedding constants do not apply directly.

---

---

#### Mode 4: `CROSS_POLYTOPE` — theoretically optimal cosine partitioning

Applies a full SRHT rotation (no subsampling), then assigns each token to its
**dominant coordinate** — the coordinate with the largest absolute value after rotation:

```python
y = H D x_padded                    # full Walsh-Hadamard rotation
j = argmax_i |y_i|                  # dominant coordinate
s = int(y_j > 0)                    # sign of dominant coordinate
partition = 2*j + s                 # in [0, 2 * padded_dim)
```

```python
from pymuvera import ProjectionType

enc = MUVERAEncoder(
  dimension=128,
  num_repetitions=4,
  projection_type=ProjectionType.CROSS_POLYTOPE,
  fill_empty_partitions=True,  # densifying fill used automatically
  seed=42,
)
# num_partitions = 2 * next_power_of_2(128) = 256  (NOT 2^k)
# fde_dimension  = 4 × 256 × 128 = 131,072
# num_simhash_projections is IGNORED for CROSS_POLYTOPE
```

**Why Cross-Polytope is theoretically superior to SimHash:** SimHash partitions space
with random hyperplanes — each bit is independent. Cross-Polytope partitions by
finding the Voronoi cell of the cross-polytope that contains the rotated vector. For
cosine similarity, Cross-Polytope cells are provably more collision-efficient: two
nearly-identical vectors are more likely to share the same dominant coordinate than
to agree on all k sign bits (Andoni & Razenshteyn, 2015).

| Model | `dimension` | `padded_dim` | `num_partitions` per rep |
|---|---|---|---|
| ColQwen2 | 128 | 128 | 256 |
| ColQwen3.5 v3 | 320 | 512 | 1,024 |

> Because `num_partitions` grows with `dimension`, enable `fill_empty_partitions=True`
> for any document corpus — densifying fill is used automatically.

---

#### Mode 5: `CALIBRATED_EIGENBASIS` — data-aware eigenbasis rotation

Rotates embeddings into the eigenbasis of the empirical token covariance before
SimHash partitioning.  Requires a one-time `calibrate(embeddings)` call on a
representative corpus sample.

```python
from pymuvera import MUVERAEncoder, ProjectionType, calibrate_from_embeddings

# Step 1: calibrate on a sample of your corpus embeddings (run once, save to disk)
calibration = calibrate_from_embeddings(corpus_embeddings)  # (N, 128) array
calibration.save("colqwen2_calibration.npz")

# Step 2: build encoder — two equivalent ways
# Option A: pass calibration at construction
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=8,
    projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
    fill_empty_partitions=True,
    seed=42,
    calibration=calibration,
)

# Option B: call calibrate() post-construction (useful when streaming corpus data)
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=8,
    projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
    fill_empty_partitions=True,
    seed=42,
)
enc.calibrate(corpus_embeddings)   # returns self, chainable

# Step 3: encode as usual — same API as every other mode
q_fde = enc.encode_query(query_tokens)
d_fde = enc.encode_document(doc_tokens)

# Step 4: load calibration in a new process without re-running the corpus pass
from pymuvera import EigenbasisCalibration
cal = EigenbasisCalibration.load("colqwen2_calibration.npz")
enc2 = MUVERAEncoder(..., projection_type=ProjectionType.CALIBRATED_EIGENBASIS, calibration=cal)
```

**How it works:** `calibrate_from_embeddings()` computes the empirical covariance
Σ of the calibration embeddings, eigendecomposes it, and stores the eigenbasis U
(eigenvectors sorted by descending eigenvalue λ).  At encode time, each embedding is
rotated into this basis (`z = x @ U`) before SimHash.  The FDE partition centroids
live in the eigenbasis space; inner products are preserved exactly since U is orthogonal.

**Eigenvalue weighting (default, `use_eigenvalue_weighting=True`):** the SimHash
projection matrix in the rotated space is sampled from N(0, diag(λ)) rather than
N(0, I).  Scaling row *i* by √λ_i makes the effective bucket-assignment metric the
λ-weighted inner product z^T diag(λ) z', concentrating discrimination on
high-variance eigenbasis coordinates.  This is the SimHash analog of the water-filling
bit allocation in SpectralQuant (Vangara & Gopinath, 2026): *structure beats budget*.

**Ablation:** set `use_eigenvalue_weighting=False` to use uniform Gaussian SimHash
in the rotated space.  This isolates the rotation contribution from the weighting
contribution at fixed compression ratio.

**Motivation:** SpectralQuant documents that LLM key covariances have participation
ratio deff/dh ≈ 3–5% at dh=128 across Qwen2.5, Mistral-7B, and Llama-3 — roughly
4–6 eigendirections carry the variance at head dimension 128.  ColQwen2 and
ColQwen3.5 are produced by trained attention heads on the same class of transformer
architecture, so the same low-effective-rank structure is expected in their patch
embeddings.  The `participation_ratio` field on `EigenbasisCalibration` lets you
verify this on your own corpus.

**Participation ratio inspection:**

```python
cal = calibrate_from_embeddings(your_embeddings)
print(f"deff = {cal.participation_ratio:.1f} / {cal.eigenvectors.shape[0]}")
# Expected for ColQwen2: deff ≈ 4–8 / 128
```

**Cost:** O(N·d²) for the rotation matmul per token, plus O(N·d·k) for SimHash —
higher than `DEFAULT_IDENTITY` at O(N·d·k).  Calibration itself (one forward pass
+ eigendecomposition) runs in seconds on a single GPU.

**Constraint:** `num_simhash_projections ≥ 1`; encoding before `calibrate()` raises
`RuntimeError`.

 — O(num_empty) fill for all projection types

By default, `fill_empty_partitions=True` uses **Hamming nearest-neighbor fill**:
for each empty slot, find the token with the smallest Hamming distance in the SimHash
sign space. This is geometrically accurate but costs O(num_tokens × k × num_empty).

**Densifying LSH fill** (Shrivastava, 2014) replaces this with a deterministic hash:

```
for each empty slot p:
    token_idx = splitmix64(p ⊕ seed) % num_tokens
    rep_slice[p] = projected[token_idx]
```

Cost: **O(num_empty)** — independent of num_tokens and k.

```python
# Explicit opt-in for sign-based modes
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=10,   # 1024 partitions — many will be empty
    num_repetitions=4,
    fill_empty_partitions=True,
    densifying_fill=True,          # O(num_empty) instead of O(N*k)
)

# Automatic for CROSS_POLYTOPE (no sketch matrix available for Hamming)
enc = MUVERAEncoder(
    dimension=320,
    num_repetitions=8,
    projection_type=ProjectionType.CROSS_POLYTOPE,
    fill_empty_partitions=True,    # densifying fill is forced automatically
    final_projection_dimension=81920,
)
```

| Fill strategy | Cost | Quality | When to use |
|---|---|---|---|
| Hamming NN (default) | O(N × k × empty) | Most geometrically precise | k ≤ 8, moderate corpus size |
| Densifying LSH | O(num_empty) | Less precise, guaranteed fill | k ≥ 10, large corpus, CROSS_POLYTOPE |

---

#### SimHash projection modes — six-way comparison (ColQwen2, d=128)

| Mode | SimHash cost (d=128) | vs baseline | Quality | Extra constraint |
|---|---|---|---|---|
| `DEFAULT_IDENTITY` | 1024N ops (k=8) | 1× | Full-rank Gaussian baseline | None |
| `LOW_RANK_GAUSSIAN` r=4 | 544N ops (k=8) | **1.9×** | Sign-pattern approx., ~25% variance ↑ | `1 ≤ r < k` |
| `LOW_RANK_GAUSSIAN` r=1 | 136N ops (k=8) | **7.5×** | ~100% variance baseline | `1 ≤ r < k` |
| `SRHT` | 904N ops (k=8) | 1.1× | Full JL, no rank error | `k ≤ next_pow2(d)` |
| `CROSS_POLYTOPE` | 896N ops (all partitions) | 1.1× | Theoretically optimal cosine | `fill` recommended |
| `CALIBRATED_EIGENBASIS` | (1024+d²)N ops (k=8) | ~0.5× | Data-aware; structure beats budget | `calibrate()` required |

#### Empty-slot fill strategies — comparison

When `fill_empty_partitions=True`, two fill strategies are available:

| Strategy | Cost | Precision | When to use |
|---|---|---|---|
| **Hamming NN** (default) | O(N × k × num_empty) | High — nearest token by SimHash distance | k ≤ 10, small–medium corpora |
| **Densifying LSH** (`densifying_fill=True`) | O(num_empty) | Lower — deterministic hash, no geometry | k ≥ 10, large corpora, `CROSS_POLYTOPE` (automatic) |

Densifying LSH fill (Shrivastava, 2014) assigns each empty slot a source token
deterministically via a splitmix64 hash of the partition index — no distance
computation, no sketch matrix required. It is **automatically used for
`CROSS_POLYTOPE`** (no sketch matrix exists for Hamming distances) and opt-in
for all other modes via `densifying_fill=True`.

**When to use each:**

* **`DEFAULT_IDENTITY`** — default choice; correctness baseline, no constraints.
* **`LOW_RANK_GAUSSIAN`** — when speed is the priority and mild quality loss is acceptable.
  **Requires k ≥ 16 and r/k ≤ 0.25** to make the tradeoff meaningful. r=4, k=6 (r/k=0.67)
  is nearly full-rank — all the variance penalty, almost no speed gain. Avoid.
* **`SRHT`** — full JL quality at sub-quadratic cost. Preferred for precision-critical
  workloads like legal/tax document retrieval where recall matters.
* **`CROSS_POLYTOPE`** — when you want theoretically optimal cosine similarity
  partitioning without tuning k. Best for high-d models (ColQwen3.5 d=320) where
  num_partitions = 2×512 = 1024 gives fine-grained coverage. Always pair with
  `fill_empty_partitions=True` (densifying fill is automatic).
* **`CALIBRATED_EIGENBASIS`** — when your corpus has a stable domain and you can
  afford a one-time calibration pass (seconds on GPU). Best for production pipelines
  on domain-specific corpora (scientific papers, legal documents, clinical notes)
  where embeddings are strongly non-isotropic. Run `calibrate_from_embeddings()` on
  a representative 256–1024 page sample; inspect `calibration.participation_ratio`
  to confirm low effective rank (< 10 for d=128 is a good signal). Save the
  calibration object alongside your FAISS index.
* **Densifying LSH fill** — when fill cost is a bottleneck (large k, large corpus),
  or whenever using `CROSS_POLYTOPE`. Enable with `densifying_fill=True` on any
  projection type. Trades geometric precision for O(num_empty) speed.

---

### Filling empty partition slots

With few document tokens and many partitions (large *k*), many slots will be
empty (all-zero). Enabling `fill_empty_partitions` copies the projection of
the nearest token by SimHash Hamming distance into each empty slot, improving
recall for short documents:

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=2,
    fill_empty_partitions=True,   # document side only; queries ignore this flag
)

short_doc_tokens = np.random.randn(8, 128).astype(np.float32)
d_fde = enc.encode_document(short_doc_tokens)   # no all-zero partition blocks
```

---

### Low-level functional API

Bypass the encoder class entirely when you need to manage parameters manually
(e.g. distributed indexing where workers share pre-built parameters):

```python
from pymuvera import FDEConfig, generate_query_fde, generate_document_fde

config = FDEConfig(
  dimension=128,
  num_repetitions=2,
  num_simhash_projections=4,
  seed=42,
)

q_fde = generate_query_fde(query_tokens, config)
d_fde = generate_document_fde(doc_tokens, config)

# Pass pre-built RepParams to skip RNG sampling on every call
enc = MUVERAEncoder(dimension=128, num_repetitions=2, num_simhash_projections=4, seed=42)
q_fde = generate_query_fde(query_tokens, config, enc._rep_params)
```

---

### `FDEConfig` serialization

`FDEConfig` is a frozen Pydantic model — save it alongside your ANN index so
the encoder configuration is always recoverable:

```python
import json
from pymuvera import FDEConfig

config = FDEConfig(dimension=128, num_repetitions=4, num_simhash_projections=4, seed=42)

# Save
with open("fde_config.json", "w") as f:
  json.dump(config.model_dump(), f)

# Load
with open("fde_config.json") as f:
  config2 = FDEConfig(**json.load(f))

assert config == config2
```

---

---

## Configuration guide

Most users hit poor results not because of a wrong projection type but because of a
misconfigured `num_simhash_projections` / `num_repetitions` / `simhash_rank` combination.
This section explains every tradeoff in plain terms, with concrete numbers for ColQwen2
(128-dim) and ColQwen3.5 (320-dim) — the two most common production models.

---

### Know your embedding dimension first

Different models produce different per-token embedding dimensions. Set `dimension` to
match your model exactly — this is the single most important parameter.

| Model | `dimension` | Notes |
|---|---|---|
| ColBERT v2 | 128 | Original late-interaction baseline |
| ColQwen2 | 128 | Most widely deployed as of 2025 |
| ColQwen3.5 v1 | 128 | Early checkpoint |
| ColQwen3.5 v3 | 320 | Current recommended checkpoint |
| Ops-ColQwen3-4B | 320 | OpenSearch variant, up to 2560 via extended head |

> **Common mistake:** Using `dimension=128` with ColQwen3.5 v3 (which is 320-dim) silently
> truncates every token embedding to 128 dims, discarding 60% of the representation before
> MUVERA even runs. Always verify with `model.config.projection_dim` or check the model card.

---

### The two knobs that matter most

#### `num_simhash_projections` (k) — partition granularity

Each repetition divides embedding space into **2^k buckets**. Tokens that land in the
same bucket get averaged together into one FDE slot.

| k | Partitions | Tokens/partition (512-token doc) | Recommendation |
|---|---|---|---|
| 4 | 16 | 32 | coarse; fast but high collision rate |
| 6 | 64 | 8 | reasonable default |
| 8 | 256 | 2 | good quality; use `fill_empty_partitions=True` |
| 10 | 1,024 | 0.5 | too sparse for most docs; many empty slots |

> **Rule of thumb:** aim for **4–10 tokens per partition** on average.
> For a 512-token ColQwen3.5 page: k=6 (8 tokens/partition) or k=8 with fill enabled.

#### `num_repetitions` — approximation quality

Each repetition is an independent random partition of the same embedding space. More
repetitions directly improves recall and is the safest quality knob to increase.

- More repetitions **always** improves recall.
- Cost scales linearly: 2× repetitions = 2× FDE size = 2× encode time.
- Diminishing returns set in around 8–16 repetitions for most corpora.

> **Rule of thumb:** start with `num_repetitions=8`. If recall is poor, double it before
> touching any other parameter.

---

### The budget equation

```
fde_dimension = num_repetitions × 2^k × dimension
```

For a fixed FDE budget, spending it on **more repetitions beats larger k** for most corpora:

| Config | fde_dimension (ColQwen3.5, d=320) | Notes |
|---|---|---|
| k=6, reps=20 | 20 × 64 × 320 = 409,600 | many repetitions, coarse partitions |
| k=8, reps=10 | 10 × 256 × 320 = 819,200 | balanced — usually better recall |
| k=8, reps=5 | 5 × 256 × 320 = 409,600 | same budget as first row; better quality |

Use `final_projection_dimension` to compress to a target index size after choosing
the right k/repetitions balance:

```python
enc = MUVERAEncoder(
    dimension=320,               # ColQwen3.5 v3
    num_simhash_projections=8,
    num_repetitions=10,
    fill_empty_partitions=True,
    final_projection_dimension=81920,  # compress to target index size
)
```

---

### When to use `fill_empty_partitions`

With k=8 (256 partitions) and a short document (< 200 tokens), many partition slots
will be empty — all zeros in the FDE. Zeros contribute nothing to the dot product and
directly hurt recall.

Enable `fill_empty_partitions=True` whenever:

```
num_doc_tokens / 2^k < 2
```

| k | Enable fill if doc tokens < |
|---|---|
| 6 | 128 |
| 8 | 512 |
| 10 | 2,048 |

For ColQwen3.5 pages at k=8: nearly always enable fill, since most document pages
produce fewer than 512 tokens.

---

### `LOW_RANK_GAUSSIAN` — when it helps and when it does not

Low-rank SimHash only makes theoretical sense when **r is much smaller than k**.
The computational benefit comes from the ratio r/k — if that ratio is close to 1,
you get all the approximation error with almost no speed gain.

| k | r | r/k ratio | Assessment |
|---|---|---|---|
| 6 | 4 | 0.67 | ❌ nearly full-rank — avoid |
| 8 | 4 | 0.50 | ⚠️ marginal benefit |
| 16 | 4 | 0.25 | ✅ good tradeoff (~1.9× faster, ~25% variance ↑) |
| 16 | 2 | 0.13 | ✅ aggressive (~4× faster, ~50% variance ↑) |

> **The k=6, rank=4 trap:** this is a near-full-rank approximation of a 6-bit matrix.
> You pay ~25% variance penalty with only a 1.4× compute saving. This combination
> produces the worst results of all modes (as seen in early ColQwen3.5 benchmarks).
> **Minimum recommended config for LOW_RANK_GAUSSIAN: k ≥ 16, rank ≤ k//4.**

---

### Recommended starting configs

#### ColQwen2 (d=128) — general purpose

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=8,
    fill_empty_partitions=True,
    seed=42,
)
# fde_dimension = 8 × 256 × 128 = 262,144
# tokens/partition at 512 tokens: 2 — fill is essential
```

#### ColQwen3.5 v3 (d=320) — general purpose

```python
enc = MUVERAEncoder(
    dimension=320,
    num_simhash_projections=8,
    num_repetitions=8,
    fill_empty_partitions=True,
    seed=42,
)
# fde_dimension = 8 × 256 × 320 = 655,360
# use final_projection_dimension if index size is a constraint
```

#### ColQwen3.5 v3 — speed-optimized (SRHT)

```python
enc = MUVERAEncoder(
    dimension=320,
    num_simhash_projections=8,
    num_repetitions=8,
    projection_type=ProjectionType.SRHT,
    fill_empty_partitions=True,
    seed=42,
)
# Full JL guarantee, ~12% faster SimHash than DEFAULT_IDENTITY at k=8
# Best quality/speed tradeoff in benchmarks
```

#### ColQwen3.5 v3 — Cross-Polytope (theoretically optimal)

```python
enc = MUVERAEncoder(
    dimension=320,
    num_repetitions=4,
    projection_type=ProjectionType.CROSS_POLYTOPE,
    fill_empty_partitions=True,    # densifying fill automatic
    seed=42,
    final_projection_dimension=81920,
)
# num_partitions = 2 * 512 = 1024 per repetition
# raw fde = 4 * 1024 * 320 = 1,310,720 -> compressed to 81,920
```

---

#### ColQwen3.5 v3 — Cross-Polytope (theoretically optimal cosine partitioning)

```python
from pymuvera import ProjectionType

enc = MUVERAEncoder(
  dimension=320,
  num_repetitions=8,
  projection_type=ProjectionType.CROSS_POLYTOPE,
  fill_empty_partitions=True,  # densifying fill used automatically — O(num_empty)
  final_projection_dimension=81920,
  seed=42,
)
# num_partitions = 2 * 512 = 1024 per repetition (next_power_of_2(320)=512)
# fde_dimension before compression = 8 × 1024 × 320 = 2,621,440
# Recommended for high-quality retrieval on complex document pages (tables, charts)
```

#### ColQwen3.5 v3 — low-rank (correctly configured)

```python
enc = MUVERAEncoder(
    dimension=320,
    num_simhash_projections=16,   # k must be large for low-rank to help
    num_repetitions=4,
    projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
    simhash_rank=4,               # r/k = 4/16 = 0.25 — meaningful low-rank
    fill_empty_partitions=True,
    seed=42,
)
# fde_dimension = 4 × 65536 × 320 = 83,886,080 — use final_projection_dimension
```

#### ColQwen2 (d=128) — calibrated eigenbasis (domain-specific corpora)

```python
from pymuvera import MUVERAEncoder, ProjectionType, calibrate_from_embeddings

# One-time calibration on a representative corpus sample
# corpus_embeddings: (N, 128) array of ColQwen2 patch embeddings from ~256 pages
calibration = calibrate_from_embeddings(corpus_embeddings)
print(f"Effective rank: {calibration.participation_ratio:.1f} / 128")
calibration.save("colqwen2_calibration.npz")

enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=8,
    projection_type=ProjectionType.CALIBRATED_EIGENBASIS,
    fill_empty_partitions=True,
    seed=42,
    calibration=calibration,
)
# fde_dimension = 8 × 256 × 128 = 262,144
# Partition assignment concentrates on the ~4–8 high-variance eigendirections
# Best for: scientific papers, legal/clinical documents, domain-specific corpora
```

--- — setting realistic expectations

MUVERA FDE retrieval is a **first-stage filter**, not a replacement for exact MaxSim.
Typical recall gaps on a 512-token ColQwen3.5 corpus:

| Stage | R@1 (typical) | Retrieval time |
|---|---|---|
| Exact MaxSim (multi-vector) | ~0.88 | slow, scales with corpus size |
| MUVERA FDE + ANN (first stage) | ~0.63 | fast, sub-linear |
| MUVERA FDE → MaxSim rerank top-100 | ~0.86 | fast + small rerank overhead |

The ~25 point R@1 gap between exact and FDE-only is normal and expected. Always pair
pymuvera with a MaxSim reranking step on the ANN shortlist for production use.

---

## Two-stage retrieval pipeline

The intended production pattern for ColQwen2 / ColBERT:

```
Offline:
  doc token embeddings  →  encode_document()  →  FDE vector  →  ANN index

Online:
  query token embeddings  →  encode_query()  →  FDE vector
                                                     │
                                              ANN search (fast, sub-linear)
                                                     │
                                            top-K candidate docs
                                                     │
                                       MaxSim re-rank on raw token embeddings
                                                     │
                                               final top-K results
```

Stage 1 (ANN on FDE vectors) eliminates 99%+ of the corpus cheaply.
Stage 2 (exact MaxSim on raw token embeddings) reranks the small candidate
set for full accuracy.

### Minimal FAISS integration

```python
import faiss
import numpy as np
from pymuvera import MUVERAEncoder

enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2, seed=42)
dim = enc.fde_dimension  # 4096

# Build index
index = faiss.IndexFlatIP(dim)  # inner product ≈ Chamfer Similarity

# Index documents (offline)
doc_embeddings = [...]  # list of (num_tokens, 128) float32 arrays
D = enc.encode_documents_batch(doc_embeddings)  # (N, 4096)
faiss.normalize_L2(D)
index.add(D)

# Query (online)
query_tokens = np.random.randn(32, 128).astype(np.float32)
q_fde = enc.encode_query(query_tokens).reshape(1, -1)
faiss.normalize_L2(q_fde)

_, candidate_ids = index.search(q_fde, k=100)  # stage 1: fast ANN
# stage 2: MaxSim re-rank candidate_ids with raw token embeddings ...
```

---

## Attribution

Python port of the C++ implementation in
[Google's graph-mining project](https://github.com/google/graph-mining/tree/main/sketching/point_cloud),
licensed under Apache 2.0.

Low-rank SimHash (`LOW_RANK_GAUSSIAN`) inspired by
[Sarkar et al., 2025](https://eshyperscale.github.io/imgs/paper.pdf),
though their theoretical results do not transfer directly to the SimHash setting.

Subsampled Randomized Hadamard Transform, (SRHT, Woolfe, Liberty, Rokhlin & Tygert, 2008)

Cross-Polytope LSH: Andoni & Razenshteyn, 2015 — *Optimal Data-Dependent Hashing for Approximate Near Neighbors*.

Densifying LSH: Shrivastava, 2014 — *Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search*.

See [NOTICE](NOTICE) for the full upstream attribution.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).