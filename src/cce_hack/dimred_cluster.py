"""PCA, K-means, HDBSCAN, UMAP on standardized mooring feature blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan

    _HAS_HDB = True
except ImportError:
    hdbscan = None
    _HAS_HDB = False

try:
    import umap

    _HAS_UMAP = True
except ImportError:
    umap = None
    _HAS_UMAP = False


def _mask_and_X(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, np.ndarray, list[str]]:
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return pd.Series([False] * len(df), index=df.index), np.empty((0, len(use))), use
    sub = df[use].replace([np.inf, -np.inf], np.nan)
    need = max(2, len(use) // 2)
    mask = sub.notna().sum(axis=1) >= need
    X = sub.loc[mask].to_numpy(dtype=float)
    return mask, X, use


def run_pca_biplot(df: pd.DataFrame, cols: list[str], n_components: int = 2) -> dict | None:
    mask, X, use = _mask_and_X(df, cols)
    if X.size == 0 or len(use) < 2:
        return None
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=min(n_components, len(use)), random_state=0)),
        ]
    )
    Z = pipe.fit_transform(X)
    n_out = Z.shape[1]
    loadings = pd.DataFrame(
        pipe.named_steps["pca"].components_.T,
        index=use,
        columns=[f"PC{i + 1}" for i in range(n_out)],
    )
    scores = pd.DataFrame(Z, columns=[f"PC{i + 1}" for i in range(n_out)])
    scores["time"] = df.loc[mask, "time"].to_numpy()
    var = pipe.named_steps["pca"].explained_variance_ratio_
    return {"scores": scores, "loadings": loadings, "explained_variance_ratio": var, "feature_names": use}


def run_kmeans(df: pd.DataFrame, cols: list[str], k: int = 4) -> pd.DataFrame | None:
    mask, X, use = _mask_and_X(df, cols)
    if X.size == 0 or len(X) < k + 10:
        return None
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("km", KMeans(n_clusters=k, random_state=0, n_init=10)),
        ]
    )
    labels = pipe.fit_predict(X)
    return pd.DataFrame({"time": df.loc[mask, "time"].to_numpy(), "regime_kmeans": labels})


def run_hdbscan(df: pd.DataFrame, cols: list[str], min_cluster_size: int = 50) -> pd.DataFrame | None:
    if not _HAS_HDB:
        return None
    mask, X, use = _mask_and_X(df, cols)
    if X.size == 0 or len(X) < min_cluster_size * 2:
        return None
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Z = pipe.fit_transform(X)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
    labels = cl.fit_predict(Z)
    return pd.DataFrame({"time": df.loc[mask, "time"].to_numpy(), "regime_hdbscan": labels})


def run_umap_2d(df: pd.DataFrame, cols: list[str], n_neighbors: int = 30) -> pd.DataFrame | None:
    if not _HAS_UMAP:
        return None
    mask, X, use = _mask_and_X(df, cols)
    if X.size == 0 or len(X) < n_neighbors * 2:
        return None
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Z = pipe.fit_transform(X)
    nn = min(n_neighbors, len(Z) - 1)
    if nn < 5:
        return None
    reducer = umap.UMAP(n_components=2, n_neighbors=nn, random_state=0, min_dist=0.1)
    emb = reducer.fit_transform(Z)
    return pd.DataFrame(
        {
            "time": df.loc[mask, "time"].to_numpy(),
            "UMAP1": emb[:, 0],
            "UMAP2": emb[:, 1],
        }
    )
