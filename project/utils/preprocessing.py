"""
Artifact loading + preprocessing for a single incoming patient profile
(e.g. from the Streamlit form). Mirrors the exact preprocessing steps used
in training/train_encoder_decoder.ipynb (Cells 13-15, 35).
"""

from pathlib import Path

import joblib
import json
import pandas as pd

from utils.constants import FEATURE_COLS


def load_preprocessing_artifacts(artifacts_dir: Path):
    """
    Load the fitted ColumnTransformer, label encoder, and config.json saved
    during training. Returns (preprocessor, label_encoder, config).
    """
    preprocessor = joblib.load(artifacts_dir / "preprocessing_pipeline.joblib")
    label_encoder = joblib.load(artifacts_dir / "label_encoder.joblib")
    with open(artifacts_dir / "config.json", "r") as f:
        config = json.load(f)
    return preprocessor, label_encoder, config


def build_input_dataframe(raw_input: dict) -> pd.DataFrame:
    """
    Turn a dict of {column_name: value} (as collected from the Streamlit
    form) into a single-row DataFrame with columns in the same order the
    model was trained on. Missing keys will raise a KeyError early, rather
    than silently producing NaNs that the imputer would mask.
    """
    missing = [c for c in FEATURE_COLS if c not in raw_input]
    if missing:
        raise ValueError(f"Missing required input field(s): {missing}")

    row = {col: raw_input[col] for col in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)


def preprocess(df: pd.DataFrame, preprocessor):
    """
    Apply the already-fitted preprocessing pipeline (imputers, scaler,
    one-hot encoder) to a raw single-row DataFrame. Returns a numpy array
    ready to be converted to a torch tensor.
    """
    return preprocessor.transform(df)
