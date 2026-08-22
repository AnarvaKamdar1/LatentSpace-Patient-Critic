"""
Streamlit interface for the MedAgent-IN obesity level encoder-decoder-classifier.

Layout: fill in patient profile fields (categorical -> dropdown, numerical ->
number input), click Predict, see the predicted class + class probabilities.
Similar-patient kNN lookup will be added in a later pass.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from utils.constants import (
    CATEGORICAL_COLS,
    CATEGORICAL_FIELDS,
    NUMERICAL_COLS,
    NUMERICAL_FIELDS,
    K_SIMILAR,
    K_HEALTHIER,
    NON_ACTIONABLE_FEATURES,
    CRITIC_TOP_N,
)
from utils.preprocessing import load_preprocessing_artifacts, build_input_dataframe, preprocess
from utils.inference import load_model, predict
from utils.similarity import (
    load_pool,
    build_latent_knn_index,
    find_similar_patients,
    get_healthier_target_class,
    find_healthier_neighbors,
)
from utils.critic import critic_analysis
from utils.auth import load_users, verify_login

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
USERS_CONFIG_PATH = BASE_DIR / "config" / "users.json"

st.set_page_config(page_title="Obesity Level Predictor", page_icon="🩺", layout="centered")


# --- Login gate ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None

if not st.session_state["authenticated"]:
    st.title("🩺 Obesity Level Predictor")
    st.subheader("Log in")

    users = load_users(USERS_CONFIG_PATH)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Log in", use_container_width=True)

    if login_submitted:
        role = verify_login(username, password, users)
        if role is None:
            st.error("Incorrect username or password.")
        else:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.rerun()

    st.stop()

role = st.session_state["role"]

with st.sidebar:
    st.write(f"Logged in as **{st.session_state['username']}**")
    st.write(f"Role: **{role.capitalize()}**")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.rerun()


@st.cache_resource
def get_artifacts():
    """Load everything once per session: preprocessor, label encoder, config, model,
    plus the latent-vector pool and a fitted latent-space kNN index."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor, label_encoder, config = load_preprocessing_artifacts(ARTIFACTS_DIR)
    model = load_model(ARTIFACTS_DIR, device)
    Z_pool, patient_profiles_pool = load_pool(ARTIFACTS_DIR)
    latent_knn_index = build_latent_knn_index(Z_pool, k=K_SIMILAR)
    return preprocessor, label_encoder, config, model, device, Z_pool, patient_profiles_pool, latent_knn_index


(
    preprocessor,
    label_encoder,
    config,
    model,
    device,
    Z_pool,
    patient_profiles_pool,
    latent_knn_index,
) = get_artifacts()

st.title("🩺 Obesity Level Predictor")
st.caption(
    f"The predictions are based on specific training data and can be incorrect."
    f"The app is aimed to help clinicians and assist them, not replace them."
    f"Regardless of the outcome, it is advised to consult to a doctor."
)

st.subheader("Patient Profile")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    raw_input = {}

    with col1:
        st.markdown("**Demographics & Habits**")
        for col in CATEGORICAL_COLS:
            field = CATEGORICAL_FIELDS[col]
            raw_input[col] = st.selectbox(
                field["label"], options=field["options"], help=field.get("help")
            )

    with col2:
        st.markdown("**Measurements & Frequencies**")
        for col in NUMERICAL_COLS:
            field = NUMERICAL_FIELDS[col]
            raw_input[col] = st.number_input(
                field["label"],
                min_value=field["min"],
                max_value=field["max"],
                value=field["default"],
                step=field["step"],
                help=field.get("help"),
            )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_df = build_input_dataframe(raw_input)
    X_proc = preprocess(input_df, preprocessor)
    pred_label, probs, latent_vector = predict(model, X_proc, label_encoder, device)

    st.divider()
    st.subheader("Prediction")
    st.success(f"Predicted class: **{pred_label.replace('_', ' ')}**")

    probs_df = (
        pd.DataFrame({"Class": list(probs.keys()), "Probability": list(probs.values())})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    probs_df["Class"] = probs_df["Class"].str.replace("_", " ")

    st.bar_chart(probs_df.set_index("Class"))
    st.dataframe(
        probs_df.style.format({"Probability": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show submitted patient profile"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.session_state["last_latent_vector"] = latent_vector

    # --- 1. Similar patients (latent-space kNN) -- doctor only --------
    if role == "doctor":
        st.divider()
        st.subheader(f"Similar Patient Profiles (top {K_SIMILAR})")
        st.caption("Nearest neighbors in the model's learned latent space, from the train+val pool.")

        similar_patients = find_similar_patients(
            latent_vector, latent_knn_index, patient_profiles_pool, k=K_SIMILAR
        )
        display_cols = ["latent_distance"] + [c for c in similar_patients.columns if c not in ("latent_distance", "target", "split")]
        st.dataframe(
            similar_patients[display_cols].style.format({"latent_distance": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # --- 2. Healthier target class + constrained kNN -------------------
    # The constrained-kNN neighbor computation always runs (both roles need
    # it, since the critic/"Recommended Changes" section below depends on
    # it) -- but the neighbor table itself is only *displayed* to doctors.
    target_class = get_healthier_target_class(pred_label)

    if target_class is None:
        st.divider()
        st.subheader("Healthier Prototype")
        st.info("Predicted class is already **Normal Weight** — no healthier prototype needed.")
    else:
        if role == "doctor":
            st.divider()
            st.subheader("Healthier Prototype")
            st.write(
                f"Predicted class **{pred_label.replace('_', ' ')}** → "
                f"one step healthier: **{target_class.replace('_', ' ')}**"
            )

        query_gender = raw_input["Gender"]
        query_age = raw_input["Age"]

        healthier_neighbors, candidate_count = find_healthier_neighbors(
            query_gender=query_gender,
            query_age=query_age,
            target_class=target_class,
            patient_profiles_pool=patient_profiles_pool,
            preprocessor=preprocessor,
            sample_proc=X_proc,
            k2=K_HEALTHIER,
        )

        if role == "doctor":
            st.caption(
                f"Candidates with Gender={query_gender}, Age in "
                f"[{query_age - 2:.1f}, {query_age + 2:.1f}], class={target_class}: {candidate_count}"
            )

            if healthier_neighbors is None:
                st.warning("No candidates satisfy the constraints — try a different profile.")
            else:
                st.markdown(f"**Top {len(healthier_neighbors)} nearest '{target_class.replace('_', ' ')}' neighbors** (same gender, age ±2):")
                neighbor_display_cols = ["feature_distance"] + [
                    c for c in healthier_neighbors.columns if c not in ("feature_distance", "target", "split")
                ]
                st.dataframe(
                    healthier_neighbors[neighbor_display_cols].style.format({"feature_distance": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

        # --- 3. Critic: recommended feature changes -- visible to both ----
        if healthier_neighbors is None:
            if role == "patient":
                st.divider()
                st.subheader("Recommended Changes")
                st.warning("Not enough comparable patient data available to generate recommendations right now.")
        else:
            st.divider()
            st.subheader("Recommended Changes")
            st.caption(
                f"Excluded non-actionable features: {', '.join(NON_ACTIONABLE_FEATURES)}. "
                f"Ranked by how much each feature would need to shift to look like a "
                f"healthier profile."
            )

            critic_df = critic_analysis(
                sample_proc=X_proc,
                sample_patient=input_df,
                healthier_neighbors=healthier_neighbors,
                preprocessor=preprocessor,
            )

            st.dataframe(
                critic_df.head(CRITIC_TOP_N).style.format({"shift_magnitude": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
