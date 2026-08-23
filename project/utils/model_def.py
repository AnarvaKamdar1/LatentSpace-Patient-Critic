"""
Model architecture definitions.

These MUST match the classes defined in training/train_encoder_decoder.ipynb
exactly (same layer shapes, same names), since we're loading a raw
state_dict (not a pickled full model) from artifacts/best_encoder_decoder_model.pth.
If the architecture here drifts from what the notebook trained, load_state_dict
will fail or silently load into the wrong shapes.
"""

import torch.nn as nn


class Encoder(nn.Module):
    """Compresses the preprocessed feature vector into a latent representation."""

    def __init__(self, input_dim, latent_dim, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Reconstructs the original feature vector from the latent representation."""

    def __init__(self, latent_dim, output_dim, hidden_dims=(64, 128)):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Classifier(nn.Module):
    """Predicts the obesity class from the latent representation."""

    def __init__(self, latent_dim, num_classes, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class EncoderDecoderClassifier(nn.Module):
    """
    Encoder-Decoder model with a classification head attached to the latent
    space. The decoder reconstructs the input features (autoencoder
    objective); the classifier head predicts the obesity level from the
    same latent representation.

    forward() returns (x_recon, logits, z) -- for inference we mainly use
    logits (for the prediction) and z (the latent vector, useful later for
    kNN similar-patient lookup).
    """

    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = Classifier(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        logits = self.classifier(z)
        return x_recon, logits, z
