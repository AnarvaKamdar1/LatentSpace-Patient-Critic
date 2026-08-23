"""
Two kNN lookups, mirroring training/train_encoder_decoder.ipynb (Cells 37, 39, 40):

1. find_similar_patients(): latent-space kNN over the full train+val pool --
   "who does this patient's latent representation look like".

2. get_healthier_target_class() + find_healthier_neighbors(): constrained
   feature-space kNN, restricted to same gender / age +-2 / one severity
   step healthier than the predicted class -- "what does a healthier but
   otherwise-similar patient look like".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils.constants import SEVERITY_ORDER, FEATURE_COLS


def load_pool(artifacts_dir: Path):
    """Load the stored train+val latent vectors and their matching cleaned
    patient profiles (row-aligned) that both kNN lookups search over."""
    Z_pool = np.load(artifacts_dir / "latent_vectors_pool.npy")
    patient_profiles_pool = pd.read_csv(artifacts_dir / "latent_patient_profiles_pool.csv")
    return Z_pool, patient_profiles_pool


def build_latent_knn_index(Z_pool: np.ndarray, k: int) -> NearestNeighbors:
    """Fit a k-NN index over the latent-vector pool once (reused across
    predictions in the same session)."""
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(Z_pool)
    return knn


def find_similar_patients(
    query_latent: np.ndarray,
    knn_index: NearestNeighbors,
    patient_profiles_pool: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Given a 1D latent vector, return the k most similar cleaned patient
    profiles from the train+val pool (by Euclidean distance in latent
    space), closest first."""
    query_latent = np.asarray(query_latent).reshape(1, -1)
    distances, indices = knn_index.kneighbors(query_latent, n_neighbors=k)
    results = patient_profiles_pool.iloc[indices[0]].copy()
    results.insert(0, "latent_distance", distances[0])
    return results.reset_index(drop=True)


def get_healthier_target_class(predicted_class: str):
    """Return the class name one severity step closer to Normal_Weight than
    `predicted_class`, or None if `predicted_class` is already Normal_Weight."""
    missing = {predicted_class} - set(SEVERITY_ORDER.keys())
    if missing:
        raise ValueError(f"SEVERITY_ORDER is missing class(es): {missing}. Update the mapping in constants.py.")

    current_rank = SEVERITY_ORDER[predicted_class]
    normal_rank = SEVERITY_ORDER["Normal_Weight"]
    if current_rank == normal_rank:
        return None
    step = -1 if current_rank > normal_rank else 1
    target_rank = current_rank + step
    return next(c for c, r in SEVERITY_ORDER.items() if r == target_rank)


def find_healthier_neighbors(
    query_gender: str,
    query_age: float,
    target_class: str,
    patient_profiles_pool: pd.DataFrame,
    preprocessor,
    sample_proc: np.ndarray,
    k2: int,
):
    """
    Constrained kNN in feature space: candidates must share the query's
    gender, be within +-2 years of age, and belong to `target_class`
    (the healthier class one severity step away). Among those candidates,
    return the k2 nearest by Euclidean distance in preprocessed feature
    space. Returns (healthier_neighbors_df_or_None, candidate_count).
    """
    candidate_mask = (
        (patient_profiles_pool["Gender"] == query_gender)
        & (patient_profiles_pool["Age"].between(query_age - 2, query_age + 2))
        & (patient_profiles_pool["NObeyesdad"] == target_class)
    )
    candidates = patient_profiles_pool[candidate_mask].copy()

    if len(candidates) == 0:
        return None, 0

    candidate_features_proc = preprocessor.transform(candidates[FEATURE_COLS])
    dists = np.linalg.norm(candidate_features_proc - sample_proc, axis=1)

    k_use = min(k2, len(candidates))
    nearest_idx = np.argsort(dists)[:k_use]

    healthier_neighbors = candidates.iloc[nearest_idx].copy()
    healthier_neighbors.insert(0, "feature_distance", dists[nearest_idx])
    healthier_neighbors = healthier_neighbors.reset_index(drop=True)

    return healthier_neighbors, len(candidates)
