# Implementation Plan: [3] CTA Clustering

**File:** `tools/cta_cluster.py`
**Depends on:** `cta_features_{kernel_id}.csv` from [1] Pass B
**Produces:** `representatives.json`

---

## Algorithm

```
1. Load per-CTA feature matrix X  (shape: n_ctas × 7)
2. Z-score normalize each feature dimension
3. Determine K range: [K_min, K_max] = [2, 20] (regular: [2,5]; irregular: [5,20])
4. For K in K_min..K_max:
     fit KMeans(n_clusters=K, n_init=10)
     compute silhouette_score
5. Select K* = argmax silhouette_score  (early-stop if score drops 2 consecutive)
6. For each cluster: select medoid (CTA with min distance to centroid)
7. weight_c = cluster_size_c / total_ctas
8. Output representatives.json
```

---

## Class Design

```python
# tools/cta_cluster.py

@dataclass
class Representative:
    x: int; y: int; z: int
    weight: float
    cluster_id: int
    cluster_size: int
    distance_to_centroid: float   # quality indicator

@dataclass
class ClusteringResult:
    kernel_id: int
    total_ctas: int
    k: int
    silhouette_score: float
    representatives: list[Representative]
    method: str   # "kmeans" or "coordinate_heuristic"


class CTAClusterer:
    def __init__(self, k_min=2, k_max=20, n_init=10, random_state=42):
        ...

    def cluster(self, kernel_id: int,
                cta_features: pd.DataFrame) -> ClusteringResult:
        """Main entry: auto-selects K, returns representatives."""

    def _select_k(self, X_norm: np.ndarray) -> tuple[int, float]:
        """Silhouette-based K selection with early stopping."""

    def _find_medoids(self, X_norm, X_orig, labels, centroids) -> list[Representative]:
        """For each cluster, find CTA closest to centroid."""

    def coordinate_heuristic_fallback(self, kernel_id: int,
                                      grid_x, grid_y, grid_z) -> ClusteringResult:
        """
        Use coordinate heuristic (corners + midpoints + interior) when
        Pass B data is unavailable or silhouette score is too low.
        Mirrors compute_sampled_ctas() logic from main.cc.
        """
```

---

## Feature Normalization

Z-score normalization per feature. Features with zero variance (e.g., all CTAs have
same L1 hit rate — common in regular kernels) are dropped from the feature matrix
before clustering to avoid degenerate distances.

```python
def normalize(X: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Returns normalized X and list of active (non-zero-variance) feature indices."""
    stds = X.std(axis=0)
    active = np.where(stds > 1e-6)[0]
    X_active = X[:, active]
    X_norm = (X_active - X_active.mean(axis=0)) / stds[active]
    return X_norm, active.tolist()
```

If fewer than 2 features have non-zero variance, fall back to coordinate heuristic.

---

## K Selection

```python
def _select_k(self, X_norm):
    best_k, best_score = 2, -1.0
    no_improve = 0
    for k in range(self.k_min, min(self.k_max + 1, len(X_norm))):
        km = KMeans(n_clusters=k, n_init=self.n_init,
                    random_state=self.random_state)
        labels = km.fit_predict(X_norm)
        if len(set(labels)) < k:   # degenerate: empty cluster
            break
        score = silhouette_score(X_norm, labels)
        if score > best_score:
            best_score, best_k = score, k
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 2:
                break
    return best_k, best_score
```

If `best_score < 0.15` (very low — CTAs are nearly identical), K=1 is used and
a single medoid represents all CTAs with weight=1.0. This is the expected result
for regular GEMM/stencil interior CTAs.

---

## Output Format

**`representatives.json`:**
```json
{
  "kernel_3": {
    "total_ctas": 1024,
    "k": 3,
    "silhouette_score": 0.61,
    "method": "kmeans",
    "representatives": [
      {"x": 0,  "y": 0, "z": 0, "weight": 0.12, "cluster_id": 0, "cluster_size": 123},
      {"x": 15, "y": 7, "z": 0, "weight": 0.63, "cluster_id": 1, "cluster_size": 645},
      {"x": 31, "y": 15,"z": 0, "weight": 0.25, "cluster_id": 2, "cluster_size": 256}
    ]
  },
  "kernel_7": {
    "total_ctas": 64,
    "k": 9,
    "silhouette_score": 0.0,
    "method": "coordinate_heuristic",
    "representatives": [
      {"x": 0, "y": 0, "z": 0, "weight": 0.111, ...},
      ...
    ]
  }
}
```

---

## Fallback Decision Tree

```
Has cta_features_{kernel_id}.csv?
  No  → coordinate_heuristic_fallback()
  Yes → run KMeans
          silhouette_score >= 0.15?
            Yes → use KMeans result
            No  → coordinate_heuristic_fallback()  (CTAs are homogeneous)
```

---

## CLI Interface

```
python tools/cta_cluster.py \
  --cta-features ./profiler_output/ \
  --kernel-ids 3,7,12 \
  --output ./representatives.json \
  [--k-min 2] [--k-max 20] \
  [--force-heuristic]      # always use coordinate heuristic (debug/baseline)
  [--plot]                 # emit per-kernel silhouette + PCA plots
```

---

## Dependencies

```
numpy >= 1.24
scikit-learn >= 1.3   # KMeans, silhouette_score
pandas >= 2.0
matplotlib            # optional, for --plot
```

---

## Notes

- The coordinate heuristic in `main.cc` (`compute_sampled_ctas`) should be kept
  in sync with `coordinate_heuristic_fallback()` here — both must produce the same
  CTA list for the same grid dimensions.
- For warp-specialized kernels (CUTLASS SM90 ex48, FA3), warp role (producer vs.
  consumer) creates bimodal CTA behavior that silhouette-based K selection will
  correctly identify as K=2. Pass B feature collection must include metrics that
  capture the producer/consumer split (e.g., different stall profiles per warp role).
- TMA-using kernels: TMA transactions may not be reflected in standard L1/L2 hit
  rate counters. Feature vector may be degenerate for TMA kernels; fall back to
  coordinate heuristic.
