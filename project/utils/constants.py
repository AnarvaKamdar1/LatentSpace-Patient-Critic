"""
Static reference data for the Streamlit input form: which columns are
categorical vs numerical (must match training/train_encoder_decoder.ipynb),
the valid dropdown options for each categorical column (from the UCI
"Estimation of Obesity Levels" dataset, id=544), and sensible min/max/default
values for each numerical column so the form doesn't accept nonsense input.
"""

CATEGORICAL_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

NUMERICAL_COLS = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# --- kNN / similar-patient + critic settings (mirrors training notebook) ---

K_SIMILAR = 5   # latent-space kNN: number of similar patients to show
K_HEALTHIER = 7  # constrained feature-space kNN: number of healthier-prototype neighbors

# Clinical/severity ordering of the obesity classes, from most underweight to
# most obese, with Normal_Weight as the "healthy" center. Moving one step
# toward Normal_Weight (in either direction) defines a "healthier" class for
# a given predicted class.
SEVERITY_ORDER = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}

# Features a patient can't change on purpose -- excluded from the critic's
# "recommended changes" table even if the healthier-prototype centroid
# differs slightly on them.
NON_ACTIONABLE_FEATURES = ["Age", "Height", "Gender", "family_history_with_overweight"]

# How many top actionable features to show in the critic table.
CRITIC_TOP_N = 8

# Dropdown options per categorical column, with a human-readable label and
# help text shown in the Streamlit form.
CATEGORICAL_FIELDS = {
    "Gender": {
        "label": "Gender",
        "options": ["Male", "Female"],
        "help": None,
    },
    "family_history_with_overweight": {
        "label": "Family history of overweight",
        "options": ["yes", "no"],
        "help": "Has a family member suffered or suffers from overweight?",
    },
    "FAVC": {
        "label": "Frequent high-caloric food (FAVC)",
        "options": ["yes", "no"],
        "help": "Do you eat high caloric food frequently?",
    },
    "CAEC": {
        "label": "Eating between meals (CAEC)",
        "options": ["no", "Sometimes", "Frequently", "Always"],
        "help": "Do you eat any food between meals?",
    },
    "SMOKE": {
        "label": "Smoker",
        "options": ["yes", "no"],
        "help": "Do you smoke?",
    },
    "SCC": {
        "label": "Monitors calorie intake (SCC)",
        "options": ["yes", "no"],
        "help": "Do you monitor the calories you eat daily?",
    },
    "CALC": {
        "label": "Alcohol consumption (CALC)",
        "options": ["no", "Sometimes", "Frequently", "Always"],
        "help": "How often do you drink alcohol?",
    },
    "MTRANS": {
        "label": "Transportation used (MTRANS)",
        "options": ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
        "help": "Which transportation do you usually use?",
    },
}

# min / max / default / step per numerical column, used for st.number_input.
NUMERICAL_FIELDS = {
    "Age": {"label": "Age (years)", "min": 14.0, "max": 65.0, "default": 25.0, "step": 1.0},
    "Height": {"label": "Height (m)", "min": 1.40, "max": 2.10, "default": 1.70, "step": 0.01},
    "Weight": {"label": "Weight (kg)", "min": 30.0, "max": 180.0, "default": 70.0, "step": 0.5},
    "FCVC": {
        "label": "Vegetable consumption frequency (FCVC)",
        "min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1,
        "help": "1 = never, 2 = sometimes, 3 = always",
    },
    "NCP": {
        "label": "Number of main meals (NCP)",
        "min": 1.0, "max": 4.0, "default": 3.0, "step": 0.1,
    },
    "CH2O": {
        "label": "Daily water intake (CH2O)",
        "min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1,
        "help": "1 = less than 1L, 2 = 1-2L, 3 = more than 2L",
    },
    "FAF": {
        "label": "Physical activity frequency (FAF)",
        "min": 0.0, "max": 3.0, "default": 1.0, "step": 0.1,
        "help": "0 = never, 1 = 1-2 days/wk, 2 = 2-4 days/wk, 3 = 4-5 days/wk",
    },
    "TUE": {
        "label": "Time using technology devices (TUE)",
        "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1,
        "help": "0-2 hrs, 3-5 hrs, >5 hrs (encoded 0-2)",
    },
}
