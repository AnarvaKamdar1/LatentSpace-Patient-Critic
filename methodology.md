# Technical Methodology

## How It Works

### 1. Prediction

The patient's profile is preprocessed through the same `ColumnTransformer` pipeline used during training, producing a 31-dimensional vector. This is fed through the Encoder to Classifier path to produce class probabilities over 7 obesity levels.

### 2. SHAP Explainability

A model-agnostic `shap.KernelExplainer` explains why the model made its prediction. It uses a k-means-summarized background sample and attributes the predicted class probability across all input features. One-hot encoded categorical features are collapsed back into single contributions per original feature for readability.

### 3. Latent-Space Similar Patient Search

*(Doctor view only)*

The encoder produces a 16-dimensional latent vector for the query patient. A pre-fitted kNN index over all training and validation latent vectors retrieves the 5 closest patients, showing their full profiles alongside their latent-space distance.

### 4. Healthier Prototype Search

*(Table shown to Doctor only; results used internally for both roles)*

The system identifies the obesity class one step closer to Normal Weight on the clinical severity scale. It then searches the patient pool for candidates matching the query's gender and age (±2 years) who belong to that healthier class, selecting the nearest neighbors by Euclidean distance in the preprocessed feature space.

### 5. Critic: Recommended Changes

*(Visible to both Doctor and Patient)*

The critic computes the centroid of the healthier prototype neighbors in the preprocessed feature space, calculates the delta from the query patient, and ranks features by shift magnitude. Non-actionable features (Age, Height, Gender, Family History) are excluded. Numeric features are shown in original units; categorical features show the most common value among the healthier neighbors with its prevalence.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Latent dimension | 16 |
| Encoder hidden layers | [128, 64] |
| Decoder hidden layers | [64, 128] |
| Classifier hidden layer | 32 |
| Dropout | 0.2 (all layers) |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Classification loss weight | 1.0 |
| Reconstruction loss weight | 0.5 |
| Max epochs | 30 |
| Early stopping patience | 15 epochs (on validation loss) |
| Data split | 70/15/15 (train/val/test, stratified) |

### Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **92.74%** |
| Best Validation Loss | 0.2660 |

---

## Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | PyTorch 2.11 |
| **ML / Preprocessing** | scikit-learn 1.6, imbalanced-learn 0.14 |
| **Explainability** | SHAP |
| **Data** | NumPy 2.0, Pandas 2.2 |
| **Visualization** | Matplotlib 3.10, Seaborn 0.13 |
| **Web Application** | Streamlit |
| **Dataset Source** | UCI ML Repository (ucimlrepo 0.0.7) |
| **Serialization** | joblib 1.5 |

