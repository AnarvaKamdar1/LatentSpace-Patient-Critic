"""
Model loading + single-sample inference. Mirrors Cell 35 of
training/train_encoder_decoder.ipynb.
"""

from pathlib import Path

import numpy as np
import torch

from utils.model_def import EncoderDecoderClassifier


def load_model(artifacts_dir: Path, device: torch.device) -> EncoderDecoderClassifier:
    """
    Reconstruct the EncoderDecoderClassifier architecture using the dims
    stored in the checkpoint itself, then load the trained weights.
    """
    checkpoint = torch.load(
        artifacts_dir / "best_encoder_decoder_model.pth",
        map_location=device,
    )

    model = EncoderDecoderClassifier(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model: EncoderDecoderClassifier, X_proc: np.ndarray, label_encoder, device: torch.device):
    """
    Run a preprocessed (already-transformed) feature array through the
    model and return:
      - pred_label: predicted class name (str)
      - probs: dict of {class_name: probability} for every class
      - latent_vector: 1D numpy array, the model's latent representation z
        (kept for later use in the kNN similar-patient lookup)
    """
    X_t = torch.tensor(X_proc, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, logits, z = model(X_t)
        probs_arr = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs_arr.argmax())
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    probs = {
        class_name: float(probs_arr[i])
        for i, class_name in enumerate(label_encoder.classes_)
    }

    latent_vector = z.cpu().numpy()[0]

    return pred_label, probs, latent_vector
