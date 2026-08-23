# Setup and Operations

## Project Structure

```text
Main Project Folder
|
|---- requirements.txt (dependencies)
|---- folder_structure.txt (this file)
|---- commands.txt (help on how to run the project)
|
|---- myenv(not included)
|       |
|       |---- Scripts -> (activate)
|       |----(other)
|       ...
|
|---- project
|       |
|       |---- app.py
|       |
|       |---- artifacts
|       |         |
|       |         |---- best_encoder_decoder_model.pth
|       |         |---- preprocessing_pipeline.joblib
|       |         |---- label_encoder.joblib
|       |         |---- config.json
|       |         |---- latent_patient_profiles_pool.csv
|       |         |---- latent_vectors_pool.npy
|       |
|       |---- training
|       |         |---- train_encoder_decoder.ipynb (original notebook)
|       |         |---- encoder_decoder.py (original notebook's python script version)
|       |
|       |---- utils
|       |         |---- __pycache__(not included)
|       |         |---- __init__.py
|       |         |---- auth.py
|       |         |---- constants.py
|       |         |---- critic.py
|       |         |---- feature_shap.py
|       |         |---- inference.py
|       |         |---- model_def.py
|       |         |---- preprocessing.py
|       |         |---- similarity.py
|       |
|       |---- config
|       |         |---- demo_users.txt (demo accounts)
|       |         |---- users.json (hashed demo passwords)
|
|---- results (contains all results as pdfs)
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Latent_Space_Patient_Critic.git
   cd Latent_Space_Patient_Critic
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv myenv

   # Windows
   myenv\Scripts\activate

   # macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
   ```

   > **Note:** The `--index-url` flag installs the CPU-only version of PyTorch to keep the download size small. If you have a CUDA-capable GPU, use the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate GPU version instead.

4. **Run the application**
   ```bash
   cd project
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`.

---

## Usage

### Running the App

After launching, you will see a login screen. Use the demo credentials below to access the app.

### Demo Accounts

| Username | Password | Role | Access Level |
|---|---|---|---|
| `doctor1` | `doctor123` | Doctor | Full access — prediction, SHAP, similar patients, healthier prototype, critic |
| `patient1` | `patient123` | Patient | Prediction, SHAP explanation, and recommended changes only |

### Workflow

1. **Log in** with a demo account
2. **Fill in the patient profile** — demographics, habits, and measurements
3. **Click "Predict"** to see:
   - Predicted obesity class with confidence probabilities
   - SHAP feature attribution bar chart
   - *(Doctor only)* Similar patient profiles from the latent space
   - *(Doctor only)* Healthier prototype patients
   - Recommended actionable changes to move toward a healthier profile

