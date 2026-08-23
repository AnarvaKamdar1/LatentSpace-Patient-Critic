"""
SHAP-based feature attribution for a single prediction.

Answers a different question than utils/critic.py:
  - critic.py:      "what should change to look like a healthier patient?"
  - feature_shap.py: "given THIS prediction, which features drove it?"

Model-agnostic (shap.KernelExplainer) rather than architecture-specific, so
this keeps working even if utils/model_def.py's architecture changes later.
KernelExplainer is explained against a small k-means-summarized background
sample from the train+val pool, and only the *predicted* class's probability
is explained (single-output regression target) to keep it fast enough for
interactive use in the Streamlit app.

Public API:
  - build_shap_background(): one-time (cacheable) setup of the background
    summary used by the explainer.
  - compute_shap_contributions(): explain one query patient's prediction,
    collapsed back to *original* features (one-hot categorical blocks are
    summed into a single row, same idea as critic.py), top-N by |contribution|.
"""

import numpy as np
import pandas as pd
import shap
import torch

from utils.constants import CATEGORICAL_COLS, NUMERICAL_COLS, FEATURE_COLS

DEFAULT_BACKGROUND_SAMPLE = 100  # rows drawn from the pool before k-means summarization
DEFAULT_BACKGROUND_CLUSTERS = 25  # k-means summary size handed to KernelExplainer
DEFAULT_NSAMPLES = 100  # KernelExplainer perturbation samples per explanation
DEFAULT_TOP_N = 6  # how many top features to return/display


def build_shap_background(
    preprocessor,
    patient_profiles_pool: pd.DataFrame,
    sample_size: int = DEFAULT_BACKGROUND_SAMPLE,
    n_clusters: int = DEFAULT_BACKGROUND_CLUSTERS,
    random_state: int = 42,
):
    """
    Build a small, representative background dataset for the SHAP
    KernelExplainer: sample up to `sample_size` rows from the train+val
    pool, preprocess them the same way a live query is preprocessed, then
    summarize with k-means down to `n_clusters` weighted points.

    This is the expensive one-time setup step -- call it once per session
    (e.g. via st.cache_resource in app.py, alongside get_artifacts()).
    Explaining a single prediction afterward (compute_shap_contributions)
    is fast.
    """
    pool = patient_profiles_pool
    if len(pool) > sample_size:
        pool = pool.sample(n=sample_size, random_state=random_state)

    background_proc = preprocessor.transform(pool[FEATURE_COLS])
    background_proc = np.asarray(background_proc, dtype=np.float64)

    n_clusters = min(n_clusters, background_proc.shape[0])
    return shap.kmeans(background_proc, n_clusters)


def _make_predict_fn(model, device: torch.device, class_idx: int):
    """
    Wrap the PyTorch model into a numpy-in/numpy-out function returning only
    the probability of `class_idx` -- i.e. the single scalar output that
    KernelExplainer will actually attribute across input features.
    """

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
        with torch.no_grad():
            _, logits, _ = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[:, class_idx]

    return predict_fn


def compute_shap_contributions(
    model,
    X_proc: np.ndarray,
    pred_label: str,
    label_encoder,
    device: torch.device,
    background_summary,
    preprocessor,
    sample_patient: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    nsamples: int = DEFAULT_NSAMPLES,
) -> pd.DataFrame:
    """
    Explain one query patient's prediction with SHAP, collapsed back to
    *original* features (one-hot categorical blocks summed into a single
    row each), sorted by |contribution| descending, top `top_n` rows.

    sample_proc:      preprocessed (1, n_features) array for the query patient.
    pred_label:       the model's predicted class name (str), as returned by
                       utils.inference.predict().
    background_summary: output of build_shap_background().
    sample_patient:   1-row DataFrame of the query patient's raw feature
                       values (e.g. the `input_df` built by
                       build_input_dataframe()), used to show human-readable
                       current values alongside each contribution.

    Returns a DataFrame with columns:
      feature, current_value, shap_value, direction
    `shap_value` is the signed SHAP contribution (in predicted-class
    probability units); `direction` is a short human-readable label of
    which way that feature pushed the prediction.
    """
    class_idx = int(np.where(label_encoder.classes_ == pred_label)[0][0])
    predict_fn = _make_predict_fn(model, device, class_idx)

    explainer = shap.KernelExplainer(predict_fn, background_summary)
    shap_values = explainer.shap_values(X_proc, nsamples=nsamples, silent=True)
    shap_values = np.asarray(shap_values).flatten()

    feature_names = preprocessor.get_feature_names_out()
    n_numeric = len(NUMERICAL_COLS)
    cat_feature_names = feature_names[n_numeric:]

    rows = []

    # --- Numeric features: SHAP value maps 1:1 to a feature -------------
    for i, col in enumerate(NUMERICAL_COLS):
        rows.append(
            {
                "feature": col,
                "current_value": sample_patient[col].iloc[0],
                "shap_value": float(shap_values[i]),
            }
        )

    # --- Categorical features: sum SHAP across the one-hot block --------
    for col in CATEGORICAL_COLS:
        col_idx_local = [
            i for i, fname in enumerate(cat_feature_names) if fname.startswith(f"cat__{col}_")
        ]
        col_idx_full = [n_numeric + i for i in col_idx_local]
        rows.append(
            {
                "feature": col,
                "current_value": sample_patient[col].iloc[0],
                "shap_value": float(np.sum(shap_values[col_idx_full])),
            }
        )

    shap_df = pd.DataFrame(rows)

    pred_label_readable = pred_label.replace("_", " ")
    shap_df["direction"] = np.where(
        shap_df["shap_value"] >= 0,
        f"toward {pred_label_readable}",
        f"away from {pred_label_readable}",
    )

    shap_df["abs_contribution"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    shap_df = shap_df.drop(columns="abs_contribution")

    return shap_df.head(top_n)
