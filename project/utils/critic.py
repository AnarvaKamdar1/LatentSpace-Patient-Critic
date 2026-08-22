"""
Critic: compare the query patient against the healthier-prototype centroid
(mean of the constrained-kNN neighbors) and report which *original*
features (not one-hot dummies) would need to shift most to move the query
closer to that healthier outcome. Mirrors Cell 41 of the training notebook.
"""

import numpy as np
import pandas as pd

from utils.constants import NON_ACTIONABLE_FEATURES, CATEGORICAL_COLS, NUMERICAL_COLS, FEATURE_COLS


def critic_analysis(
    sample_proc: np.ndarray,
    sample_patient: pd.DataFrame,
    healthier_neighbors: pd.DataFrame,
    preprocessor,
) -> pd.DataFrame:
    """
    sample_proc: preprocessed (1, n_features) array for the query patient.
    sample_patient: 1-row DataFrame of the query patient's raw feature values.
    healthier_neighbors: output of find_healthier_neighbors() (must include
        the original FEATURE_COLS columns, not just distances).
    preprocessor: the fitted ColumnTransformer used for training.

    Returns a DataFrame with one row per actionable feature (numeric or
    categorical), sorted by how much it would need to shift, descending.
    """
    neighbor_features_proc = preprocessor.transform(healthier_neighbors[FEATURE_COLS])
    prototype_centroid_proc = neighbor_features_proc.mean(axis=0, keepdims=True)

    delta_proc = (prototype_centroid_proc - sample_proc).flatten()

    feature_names = preprocessor.get_feature_names_out()
    n_numeric = len(NUMERICAL_COLS)

    # Inverse-transform the numeric block back to original units for readability
    num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    sample_num_orig = num_scaler.inverse_transform(sample_proc[:, :n_numeric])[0]
    centroid_num_orig = num_scaler.inverse_transform(prototype_centroid_proc[:, :n_numeric])[0]

    critic_rows = []

    # --- Numeric features: one row each, values in original units ---
    for i, col in enumerate(NUMERICAL_COLS):
        if col in NON_ACTIONABLE_FEATURES:
            continue
        critic_rows.append(
            {
                "feature": col,
                "current_value": round(float(sample_num_orig[i]), 2),
                "prototype_value": round(float(centroid_num_orig[i]), 2),
                "shift_magnitude": round(abs(float(delta_proc[i])), 3),
            }
        )

    # --- Categorical features: collapse each one-hot block back into a
    # single row, comparing the query's actual category against the most
    # common category among the healthier-prototype neighbors ---
    cat_feature_names = feature_names[n_numeric:]
    for col in CATEGORICAL_COLS:
        if col in NON_ACTIONABLE_FEATURES:
            continue
        col_idx_local = [i for i, fname in enumerate(cat_feature_names) if fname.startswith(f"cat__{col}_")]
        col_idx_full = [n_numeric + i for i in col_idx_local]

        current_category = sample_patient[col].iloc[0]

        mode_counts = healthier_neighbors[col].value_counts(normalize=True)
        prototype_category = mode_counts.index[0]
        prototype_share = mode_counts.iloc[0]

        shift_magnitude = float(np.linalg.norm(delta_proc[col_idx_full]))

        critic_rows.append(
            {
                "feature": col,
                "current_value": current_category,
                "prototype_value": f"{prototype_category} ({prototype_share:.0%} of neighbors)",
                "shift_magnitude": round(shift_magnitude, 3),
            }
        )

    critic_df = pd.DataFrame(critic_rows)
    critic_df = critic_df.sort_values("shift_magnitude", ascending=False).reset_index(drop=True)
    return critic_df
