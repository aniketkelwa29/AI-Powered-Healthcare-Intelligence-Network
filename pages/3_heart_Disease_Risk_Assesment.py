# mediassist_heart_assessment_clean.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import plotly.express as px
import shap
import warnings
from pathlib import Path
from lightgbm import LGBMClassifier

warnings.simplefilter(action="ignore", category=UserWarning)

# -----------------------
# Utility functions
# -----------------------
def safe_load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pkl.load(f)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

def select_with_placeholder(label, options, key=None, placeholder="Select..."):
    opts = [placeholder] + options
    return st.selectbox(label, opts, index=0, key=key)

def find_expected_raw_cols(enc):
    """
    Best-effort to detect the encoder's expected raw columns (before encoding).
    Many category-encoders have `.cols` or `.feature_names_in_`.
    """
    try:
        if hasattr(enc, "cols") and getattr(enc, "cols") is not None:
            return list(enc.cols)
    except Exception:
        pass
    try:
        if hasattr(enc, "feature_names_in_") and getattr(enc, "feature_names_in_") is not None:
            return list(enc.feature_names_in_)
    except Exception:
        pass
    # fallback to common full set from this project (22 columns)
    return [
        'gender', 'race', 'general_health', 'health_care_provider',
        'could_not_afford_to_see_doctor', 'length_of_time_since_last_routine_checkup',
        'ever_diagnosed_with_heart_attack', 'ever_diagnosed_with_a_stroke',
        'ever_told_you_had_a_depressive_disorder', 'ever_told_you_have_kidney_disease',
        'ever_told_you_had_diabetes', 'BMI', 'difficulty_walking_or_climbing_stairs',
        'physical_health_status', 'mental_health_status', 'asthma_Status',
        'smoking_status', 'binge_drinking_status', 'exercise_status_in_past_30_Days',
        'age_category', 'sleep_category', 'drinks_category'
    ]

def ensure_df_from_encoded(encoded, encoder, model=None):
    """Return a DataFrame from encoder.transform output; try to infer columns."""
    if isinstance(encoded, pd.DataFrame):
        return encoded.copy()
    arr = np.asarray(encoded)
    cols = None
    try:
        if hasattr(encoder, "get_feature_names_out"):
            cols = list(encoder.get_feature_names_out())
    except Exception:
        cols = None
    if cols is None and model is not None and hasattr(model, "feature_names_in_"):
        try:
            cols = list(model.feature_names_in_)
        except Exception:
            cols = None
    if cols is None:
        cols = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)

def find_lgbm(est):
    try:
        # if ensemble or wrapper, try to find underlying LGBM
        candidate = est
        if hasattr(est, "estimators_") and len(est.estimators_) > 0:
            candidate = est.estimators_[0]
        if hasattr(candidate, "steps"):
            for _, s in reversed(candidate.steps):
                if isinstance(s, LGBMClassifier):
                    return s
                if hasattr(s, "predict_proba") and hasattr(s, "feature_importances_"):
                    return s
        if isinstance(candidate, LGBMClassifier):
            return candidate
    except Exception:
        pass
    return None

# -----------------------
# Paths & load model/encoder
# -----------------------
MODEL_PATH = Path("models/third_feature_models/best_model.pkl")
ENCODER_PATH = Path("models/third_feature_models/cbe_encoder.pkl")

if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    st.error("Model or encoder not found. Put best_model.pkl and cbe_encoder.pkl into models/third_feature_models/")
    st.stop()

model = safe_load_pickle(MODEL_PATH)
encoder = safe_load_pickle(ENCODER_PATH)

# Determine expected raw columns (order matters for encoder)
EXPECTED_RAW_COLS = find_expected_raw_cols(encoder)

# Sensible defaults (used if user leaves optional blank)
DEFAULTS = {
    'gender': 'male',
    'race': 'white_only_non_hispanic',
    'general_health': 'good',
    'health_care_provider': 'yes_only_one',
    'could_not_afford_to_see_doctor': 'no',
    'length_of_time_since_last_routine_checkup': 'past_year',
    'ever_diagnosed_with_heart_attack': 'no',
    'ever_diagnosed_with_a_stroke': 'no',
    'ever_told_you_had_a_depressive_disorder': 'no',
    'ever_told_you_have_kidney_disease': 'no',
    'ever_told_you_had_diabetes': 'no',
    'BMI': 'normal_weight_bmi_18_5_to_24_9',
    'difficulty_walking_or_climbing_stairs': 'no',
    'physical_health_status': 'zero_days_not_good',
    'mental_health_status': 'zero_days_not_good',
    'asthma_Status': 'never_asthma',
    'smoking_status': 'never_smoked',
    'binge_drinking_status': 'no',
    'exercise_status_in_past_30_Days': 'yes',
    'age_category': 'Age_40_to_44',
    'sleep_category': 'normal_sleep_6_to_8_hours',
    'drinks_category': 'did_not_drink'
}

# -----------------------
# Mapping dictionaries (user-friendly -> model labels)
# -----------------------
age_map = {
    "18–24 years": "Age_18_to_24",
    "25–29 years": "Age_25_to_29",
    "30–34 years": "Age_30_to_34",
    "35–39 years": "Age_35_to_39",
    "40–44 years": "Age_40_to_44",
    "45–49 years": "Age_45_to_49",
    "50–54 years": "Age_50_to_54",
    "55–59 years": "Age_55_to_59",
    "60–64 years": "Age_60_to_64",
    "65–69 years": "Age_65_to_69",
    "70–74 years": "Age_70_to_74",
    "75–79 years": "Age_75_to_79",
    "80+ years": "Age_80_or_older"
}

bmi_map = {
    "Underweight (BMI < 18.5)": "underweight_bmi_less_than_18_5",
    "Normal (18.5 – 24.9)": "normal_weight_bmi_18_5_to_24_9",
    "Overweight (25 – 29.9)": "overweight_bmi_25_to_29_9",
    "Obese (30+)": "obese_bmi_30_or_more"
}

gender_map = {"Male": "male", "Female": "female", "Non-binary": "nonbinary"}

yesno_map = {"Yes": "yes", "No": "no"}

smoke_map = {"Never smoked": "never_smoked", "Used to smoke": "former_smoker", "Currently smoke": "current_smoker_every_day"}

sleep_map = {
    "0–3 hours": "very_short_sleep_0_to_3_hours",
    "4–5 hours": "short_sleep_4_to_5_hours",
    "6–8 hours (normal)": "normal_sleep_6_to_8_hours",
    "9+ hours": "long_sleep_9_to_10_hours"
}

alcohol_map = {"I don't drink": "did_not_drink", "1–7 drinks/week": "very_low_consumption_0.01_to_1_drinks", "8+ drinks/week": "high_consumption_10.01_to_20_drinks"}

general_health_map = {"Excellent": "excellent", "Very good": "very_good", "Good": "good", "Fair": "fair", "Poor": "poor"}

# -----------------------
# Page style & layout
# -----------------------
try:
    icon = Image.open("utils/heart_disease.jpg")
except Exception:
    icon = None

st.set_page_config(page_title="MediAssist — Heart Disease Risk", layout="wide", page_icon=icon)
st.markdown("""
    <style>
      /* Clean dark-card look */
      .card { background: linear-gradient(90deg, #0b0f1a, #0f1724); padding: 18px; border-radius: 12px; box-shadow: rgba(0,0,0,0.4) 0px 4px 20px; color: #e6eef8 }
      .muted { color: #9aa7b2; font-size: 0.95rem }
      .big-btn { display:block; width:100%; padding:12px; border-radius:10px; background:linear-gradient(90deg,#ff6b6b,#ff8a5c); color:white; font-weight:600; border:none; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("utils/ph5.png", use_container_width=True)
st.sidebar.header("About MediAssist")
st.sidebar.write("AI-based heart disease risk assessment. This is educational only — consult a doctor for medical advice.")

# Title
st.title("🩺 MediAssist — Quick Heart Disease Risk Check")
st.write("Answer a few simple questions — it only takes ~1 minute. The app will show a risk percentage, the main contributing factors, and easy-to-follow recommendations.")
st.write("---")

# -----------------------
# Inputs (user-friendly, placeholder-first)
# -----------------------
left, right = st.columns([2, 3])

with left:
    st.header("👤 About you")
    gender_sel = select_with_placeholder("What is your gender?", ["Male", "Female", "Non-binary"], key="gender_sel")
    age_sel = select_with_placeholder("What is your age range?", list(age_map.keys()), key="age_sel")
    health_sel = select_with_placeholder("How would you rate your overall health?", list(general_health_map.keys()), key="health_sel")

    st.write("")  # spacing
    st.header("🩺 Medical")
    diabetes_sel = select_with_placeholder("Have you been told you have diabetes?", ["Yes", "No"], key="diabetes_sel")
    heartattack_sel = select_with_placeholder("Have you had a heart attack before?", ["Yes", "No"], key="heartattack_sel")
    cholesterol_sel = select_with_placeholder("Do you have high cholesterol (doctor told you)?", ["Yes", "No"], key="chol_sel")

with right:
    st.header("🏃 Lifestyle")
    bmi_sel = select_with_placeholder("Which best describes your weight?", list(bmi_map.keys()), key="bmi_sel")
    smoking_sel = select_with_placeholder("Do you smoke?", ["Never smoked", "Used to smoke", "Currently smoke"], key="smoke_sel")
    exercise_sel = select_with_placeholder("Did you do physical activity/exercise in the past 30 days?", ["Yes", "No"], key="exercise_sel")
    sleep_sel = select_with_placeholder("How many hours of sleep do you usually get?", list(sleep_map.keys()), key="sleep_sel")
    alcohol_sel = select_with_placeholder("Do you drink alcohol?", list(alcohol_map.keys()), key="alcohol_sel")

# Advanced optional collected in expander (keeps original fields available)
with st.expander("Advanced (optional) — more details for a finer score"):
    colA1, colA2 = st.columns(2)
    with colA1:
        race_sel = select_with_placeholder("Your race/ethnicity", ["white_only_non_hispanic", "black_only_non_hispanic", "hispanic", "asian_only_non_hispanic"], key="race_sel")
        walking_sel = select_with_placeholder("Difficulty walking/climbing stairs?", ["Yes", "No"], key="walk_sel")
    with colA2:
        kidney_sel = select_with_placeholder("Ever told you have kidney disease?", ["Yes", "No"], key="kidney_sel")
        depression_sel = select_with_placeholder("Ever told you have depressive disorder?", ["Yes", "No"], key="dep_sel")

# -----------------------
# Prepare the raw input (map friendly -> model labels, and fill defaults)
# -----------------------
def map_choice(value, mapping, default_key):
    if not isinstance(value, str) or value.strip() == "Select...":
        return DEFAULTS.get(default_key)
    return mapping.get(value, value)  # if already a model label, pass through

def build_raw_input():
    # Map friendly selections into dataset labels
    raw = {}
    # Demographics & medical
    raw['gender'] = map_choice(gender_sel, gender_map, 'gender')
    raw['age_category'] = map_choice(age_sel, age_map, 'age_category')
    raw['general_health'] = map_choice(health_sel, general_health_map, 'general_health')
    raw['ever_told_you_had_diabetes'] = map_choice(diabetes_sel, yesno_map, 'ever_told_you_had_diabetes')
    raw['ever_diagnosed_with_heart_attack'] = map_choice(heartattack_sel, yesno_map, 'ever_diagnosed_with_heart_attack')
    raw['ever_told_you_have_kidney_disease'] = map_choice(kidney_sel if 'kidney_sel' in locals() else None, yesno_map, 'ever_told_you_have_kidney_disease')
    raw['ever_told_you_had_a_depressive_disorder'] = map_choice(depression_sel if 'dep_sel' in locals() else None, yesno_map, 'ever_told_you_had_a_depressive_disorder')
    # Lifestyle
    raw['BMI'] = map_choice(bmi_sel, bmi_map, 'BMI')
    raw['smoking_status'] = map_choice(smoking_sel, smoke_map, 'smoking_status')
    raw['exercise_status_in_past_30_Days'] = map_choice(exercise_sel, yesno_map, 'exercise_status_in_past_30_Days')
    raw['sleep_category'] = map_choice(sleep_sel, sleep_map, 'sleep_category')
    raw['drinks_category'] = map_choice(alcohol_sel, alcohol_map, 'drinks_category')
    raw['race'] = map_choice(race_sel if 'race_sel' in locals() else None, {}, 'race')
    raw['difficulty_walking_or_climbing_stairs'] = map_choice(walking_sel if 'walk_sel' in locals() else None, yesno_map, 'difficulty_walking_or_climbing_stairs')
    # Fill all expected raw columns in correct order with defaults where needed
    df = pd.DataFrame([raw])
    for col in EXPECTED_RAW_COLS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, "")
    # reorder
    df = df[EXPECTED_RAW_COLS]
    # Replace any placeholder-ish values
    for col in df.columns:
        v = df.at[0, col]
        if isinstance(v, str) and v.strip() in ("", "Select...", "None"):
            df.at[0, col] = DEFAULTS.get(col, "")
    return df

# -----------------------
# Predict button & logic
# -----------------------
st.write("")  # spacer
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    run_btn = st.button("🚀 Get My Risk Assessment", use_container_width=True)

if run_btn:
    # basic check: ensure minimal required fields (these are user-friendly ones)
    missing = []
    required_pairs = {
        "Gender": gender_sel, "Age": age_sel, "Weight category": bmi_sel,
        "Smoking": smoking_sel, "Exercise": exercise_sel, "General health": health_sel
    }
    for label, val in required_pairs.items():
        if not isinstance(val, str) or val.strip() == "Select...":
            missing.append(label)
    if missing:
        st.error("Please answer: " + ", ".join(missing) + " (top-left)")
    else:
        input_raw_df = build_raw_input()
        st.markdown("**Preparing input…**")
        try:
            # transform with encoder (defensive)
            enc_out = encoder.transform(input_raw_df)
            encoded_df = ensure_df_from_encoded(enc_out, encoder, model)
            # Align encoded_df with model if possible
            if hasattr(model, "feature_names_in_"):
                encoded_df = encoded_df.reindex(columns=list(model.feature_names_in_), fill_value=0)
        except Exception as e:
            # try to ensure we send exactly EXPECTED_RAW_COLS to encoder
            try:
                input_raw_df = input_raw_df[EXPECTED_RAW_COLS]
                enc_out = encoder.transform(input_raw_df)
                encoded_df = ensure_df_from_encoded(enc_out, encoder, model)
                if hasattr(model, "feature_names_in_"):
                    encoded_df = encoded_df.reindex(columns=list(model.feature_names_in_), fill_value=0)
            except Exception as e2:
                st.error(f"Encoder transform failed: {e2}")
                st.stop()

        # prediction
        try:
            proba = model.predict_proba(encoded_df)[:, 1][0] * 100.0
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        # nice result card
        risk_level = "Low"
        color = "#16a34a"
        if proba > 70:
            risk_level = "Very High"
            color = "#dc2626"
        elif proba > 40:
            risk_level = "High"
            color = "#f97316"
        elif proba > 25:
            risk_level = "Moderate"
            color = "#f59e0b"

        st.markdown(f"""
            <div class="card">
              <h2 style="margin:0;color:{color}">Risk: {proba:.1f}% — {risk_level}</h2>
              <p class="muted">This is an AI estimate. Always consult a medical professional for diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)

        # compute feature contributions (SHAP or fallback)
        lgbm = find_lgbm(model)
        feat_df = None
        if lgbm is not None:
            try:
                explainer = shap.TreeExplainer(lgbm)
                shap_vals = explainer.shap_values(encoded_df)
                if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                    sv = np.asarray(shap_vals[1])
                else:
                    sv = np.asarray(shap_vals)
                abs_mean = np.mean(np.abs(sv), axis=0)
                total = abs_mean.sum()
                if total == 0 or np.isnan(total):
                    abs_pct = np.ones_like(abs_mean) / len(abs_mean) * 100
                else:
                    abs_pct = (abs_mean / total) * 100
                feat_df = pd.DataFrame({"Feature": encoded_df.columns, "Importance": np.round(abs_pct, 2)}).sort_values("Importance", ascending=False)
            except Exception:
                feat_df = None

        # fallback to model.feature_importances_
        if (feat_df is None or feat_df.empty) and hasattr(lgbm, "feature_importances_"):
            try:
                fi = np.array(lgbm.feature_importances_, dtype=float)
                total = fi.sum()
                if total == 0:
                    fi_pct = np.ones_like(fi) / len(fi) * 100
                else:
                    fi_pct = (fi / total) * 100
                feat_df = pd.DataFrame({"Feature": encoded_df.columns, "Importance": np.round(fi_pct, 2)}).sort_values("Importance", ascending=False)
            except Exception:
                feat_df = None

        # Show top contributors chart and friendly recommendations
        if feat_df is not None and not feat_df.empty:
            topn = min(6, feat_df.shape[0])
            top_df = feat_df.head(topn).copy()
            other = max(0.0, 100 - top_df['Importance'].sum())
            # replaced deprecated append() with pd.concat()
            other_row = pd.DataFrame([{"Feature": "Other Factors", "Importance": np.round(other, 2)}])
            top_df2 = pd.concat([top_df, other_row], ignore_index=True)

            st.markdown("### What contributed most to this score")
            fig = px.pie(top_df2, names="Feature", values="Importance", title="Top contributors")
            st.plotly_chart(fig, use_container_width=True)

            # Simple natural language recommendations (based on top features + user answers)
            recs = []
            if proba > 70:
                recs.append("Your risk is very high — please consult a healthcare professional promptly.")
            elif proba > 40:
                recs.append("Your risk is high — schedule a medical checkup and focus on these lifestyle changes.")
            elif proba > 25:
                recs.append("Your risk is moderate — small changes in diet and activity can help reduce risk.")
            else:
                recs.append("Your risk is low — keep maintaining healthy habits!")

            top_features = list(top_df['Feature'].head(5).values)
            # friendly checks:
            if "ever_told_you_had_diabetes" in encoded_df.columns and input_raw_df.at[0, 'ever_told_you_had_diabetes'] == "yes":
                recs.append("- Manage diabetes well: medication adherence, diet, and glucose monitoring.")
            if "ever_diagnosed_with_heart_attack" in encoded_df.columns and input_raw_df.at[0, 'ever_diagnosed_with_heart_attack'] == "yes":
                recs.append("- Prior heart attack: regular follow-up with cardiologist and medication adherence.")
            # smoking check: input may contain 'current_smoker_every_day' so look for 'current_smoker'
            if "smoking_status" in encoded_df.columns and ("current_smoker" in input_raw_df.at[0, 'smoking_status']):
                recs.append("- Quitting smoking will substantially reduce heart risk.")
            if "exercise_status_in_past_30_Days" in encoded_df.columns and input_raw_df.at[0, 'exercise_status_in_past_30_Days'] == "no":
                recs.append("- Increase physical activity: aim for 30 minutes most days.")
            if "BMI" in encoded_df.columns and ("obese" in input_raw_df.at[0, 'BMI'] or "overweight" in input_raw_df.at[0, 'BMI']):
                recs.append("- Weight management: balanced diet & exercise can help lower risk.")
            if "sleep_category" in encoded_df.columns and input_raw_df.at[0, 'sleep_category'] in ("very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours"):
                recs.append("- Improve sleep: aim for consistent 6–8 hours/night.")

            # deduplicate
            final = []
            seen = set()
            for r in recs:
                if r not in seen:
                    final.append(r); seen.add(r)

            st.markdown("### Recommendations")
            for r in final:
                st.markdown(f"- {r}")
        else:
            st.info("Could not compute contributors; showing only risk and general recommendations.")
            if proba > 50:
                st.markdown("- High risk: see a doctor.")
            elif proba > 25:
                st.markdown("- Moderate risk: consider lifestyle changes.")
            else:
                st.markdown("- Low risk: maintain healthy habits.")

        st.markdown("---")
