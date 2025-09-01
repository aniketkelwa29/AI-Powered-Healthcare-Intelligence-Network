import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast
import re
from pathlib import Path

# Optional fuzzy matcher
try:
    from thefuzz import process as fuzz_process
except Exception:
    fuzz_process = None

# ----------------------
# Streamlit config & style
# ----------------------
st.set_page_config(page_title="🩺 MediAssist - AI Disease Predictor", page_icon="🩺", layout="wide")
st.markdown("""
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .block-container {padding: 1rem 2rem;}
  .small {font-size:0.9rem;color:#889}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Utilities
# ----------------------
NORMALIZE_RX = re.compile(r"[^a-z0-9]+")

def norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = NORMALIZE_RX.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

@st.cache_data(show_spinner=False)
def find_base_paths() -> list:
    candidates = [
        Path("data/Disease-Prediction-and-Medical dataset"),
        Path("data/medical"),
        Path("data"),
        Path("."),
    ]
    return [p for p in candidates if p.exists()]

@st.cache_data(show_spinner=False)
def safe_read_csv(name: str) -> pd.DataFrame:
    for base in find_base_paths():
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    if Path(name).exists():
        return pd.read_csv(name)
    raise FileNotFoundError(f"Could not find '{name}'. Checked: " + ", ".join(str(base / name) for base in find_base_paths() + [Path(name)]))

@st.cache_resource(show_spinner=False)
def load_resources():
    sym_des = safe_read_csv("symptoms_df.csv")
    precautions = safe_read_csv("precautions_df.csv")
    workout = safe_read_csv("workout_df.csv")
    description = safe_read_csv("description.csv")
    medications = safe_read_csv("medications.csv")
    diets = safe_read_csv("diets.csv")

    model_paths = [
        Path("models/first_feature_models/RandomForest.pkl"),
        Path("models/RandomForest.pkl"),
        Path("RandomForest.pkl"),
        Path("models/model.pkl"),
    ]
    model = None
    for mp in model_paths:
        if mp.exists():
            with open(mp, "rb") as f:
                model = pickle.load(f)
            break
    if model is None:
        raise FileNotFoundError("RandomForest model file not found in expected locations.")

    return sym_des, precautions, workout, description, medications, diets, model

# ----------------------
# Feature helpers
# ----------------------
def extract_feature_names(sym_des: pd.DataFrame, model) -> list:
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in list(model.feature_names_in_)]
    cols = [c for c in sym_des.columns if norm(c) != "disease"]
    return cols

@st.cache_data(show_spinner=False)
def build_alias_map(feature_names: list) -> dict:
    alias = {}
    synonyms = {
        "fever": "high_fever",
        "high fever": "high_fever",
        "stomach pain": "abdominal_pain",
        "belly pain": "abdominal_pain",
        "tummy pain": "abdominal_pain",
        "throat pain": "sore_throat",
        "runny nose": "runny_nose",
        "blocked nose": "congestion",
        "body ache": "muscle_pain",
        "loose motion": "diarrhoea",
        "vomitting": "vomiting",
        "coughing": "cough",
        "cold": "chills",
        "bp": "high_blood_pressure",
        "sugar": "high_blood_sugar",
        "breathlessness": "shortness_of_breath",
        "chest pain": "chest_pain",
        "itch": "itching",
        "period pain": "menstrual_pain",
    }

    for feat in feature_names:
        c = str(feat)
        n = norm(c)
        alias[n] = c
        alias[n.replace(" ", "_")] = c
        alias[n.replace(" ", "")] = c
        alias[n.replace("_", " ")] = c

    for k, v in list(synonyms.items()):
        if v in feature_names:
            alias[norm(k)] = v

    return alias


def match_symptom(token: str, alias_map: dict, threshold: int = 86):
    key = norm(token)
    if key in alias_map:
        return alias_map[key], key, 100

    if fuzz_process is not None:
        choices = list(alias_map.keys())
        match = fuzz_process.extractOne(key, choices)
        if match:
            best_key, score = match[0], match[1]
            if score >= threshold:
                return alias_map[best_key], best_key, score
            else:
                return alias_map[best_key], best_key, score
    return None, None, 0


def parse_free_text(text: str) -> list:
    if not text:
        return []
    text = text.replace("/", ",").replace(";", ",")
    text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)
    raw = [t.strip() for t in text.split(",")]
    return [t for t in raw if t]


def symptoms_to_vector(selected_features: list, feature_names: list) -> np.ndarray:
    vec = np.zeros(len(feature_names), dtype=float)
    name_to_idx = {f: i for i, f in enumerate(feature_names)}
    for f in selected_features:
        i = name_to_idx.get(f)
        if i is not None:
            vec[i] = 1.0
    return vec

# ----------------------
# Load resources
# ----------------------
try:
    sym_des, precautions, workout, description, medications, diets, model = load_resources()
except Exception as e:
    st.error(f"❌ Error loading resources: {e}")
    st.stop()

FEATURES = extract_feature_names(sym_des, model)
ALIAS_MAP = build_alias_map(FEATURES)

# Build normalized disease maps from description dataframe (preferred source of truth)
norm2orig = {}
disease_description = {}
try:
    # find disease column
    desc_cols = [c for c in description.columns if 'disease' in c.lower()]
    desc_col = desc_cols[0] if desc_cols else description.columns[0]
    desc_text_col = next((c for c in description.columns if 'description' in c.lower()), None)
    for _, row in description.iterrows():
        orig = row.get(desc_col, None)
        if pd.isna(orig) or orig is None:
            continue
        orig = str(orig).strip()
        k = norm(orig)
        norm2orig[k] = orig
        if desc_text_col:
            v = row.get(desc_text_col, '')
            disease_description[k] = str(v) if not pd.isna(v) else ''
        else:
            # try second column
            others = [c for c in description.columns if c != desc_col]
            disease_description[k] = str(row.get(others[0], '')) if others else ''
except Exception:
    norm2orig = {}
    disease_description = {}

# Helper to parse various cell formats into list of strings

def parse_cell_to_list(raw):
    items = []
    if pd.isna(raw) or raw is None:
        return items
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if pd.notna(x) and str(x).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        # literal list-like
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        # comma/semicolon/newline separated
        if any(delim in s for delim in [',', ';', '\n']):
            parts = re.split(r'[,;\n]+', s)
            return [p.strip() for p in parts if p.strip()]
        # otherwise single value
        return [s]
    # fallback
    try:
        return [str(raw).strip()]
    except Exception:
        return []

# Build disease -> lists dictionaries from dataframes

def build_lookup_from_df(df, value_cols=None):
    mapping = {}
    if df is None or df.empty:
        return mapping
    # find disease column
    disease_cols = [c for c in df.columns if 'disease' in c.lower()]
    key_col = disease_cols[0] if disease_cols else df.columns[0]
    # identify which columns to treat as values
    if value_cols:
        cols_to_use = [c for c in value_cols if c in df.columns]
    else:
        cols_to_use = [c for c in df.columns if c != key_col]
    for _, row in df.iterrows():
        raw_d = row.get(key_col, None)
        if pd.isna(raw_d) or raw_d is None:
            continue
        orig = str(raw_d).strip()
        k = norm(orig)
        norm2orig.setdefault(k, orig)
        items = []
        for c in cols_to_use:
            raw = row.get(c, None)
            items.extend(parse_cell_to_list(raw))
        # deduplicate preserving order
        seen = set()
        out = []
        for it in items:
            if it and it not in seen:
                seen.add(it)
                out.append(it)
        mapping[k] = out
    return mapping

# Build the specific mappings
try:
    # For precautions, many datasets put each precaution in its own column after 'Disease'
    disease_precautions = build_lookup_from_df(precautions)
except Exception:
    disease_precautions = {}

try:
    # Medications: often in a column called 'Medication' or similar
    med_candidates = [c for c in medications.columns if 'med' in c.lower()]
    disease_medications = build_lookup_from_df(medications, value_cols=med_candidates if med_candidates else None)
except Exception:
    disease_medications = {}

try:
    diet_candidates = [c for c in diets.columns if 'diet' in c.lower()]
    disease_diets = build_lookup_from_df(diets, value_cols=diet_candidates if diet_candidates else None)
except Exception:
    disease_diets = {}

try:
    # workouts/exercises
    workout_cols = [c for c in workout.columns if c.lower() != 'disease']
    disease_exercises = build_lookup_from_df(workout)
except Exception:
    disease_exercises = {}

# Build a canonical list of diseases (original names) for searching
DISEASE_ORIG_LIST = [norm2orig[k] for k in norm2orig.keys()]
DISEASE_NORM_LIST = list(norm2orig.keys())

# ----------------------
# Sidebar & UI
# ----------------------
st.sidebar.image("utils/ph3.png", use_container_width=True)
st.sidebar.markdown("### 🤖 About MediAssist")
st.sidebar.info("AI-powered disease prediction for educational purposes. Not medical advice.")

with st.sidebar.expander("⚙️ Settings", expanded=False):
    fuzzy_threshold = st.slider("Fuzzy match threshold", 70, 95, 86, 1)
    min_symptoms_needed = st.number_input("Minimum symptoms to predict", 1, 10, 2)
    min_prob_warn = st.slider("Warn if top probability below (%)", 0, 100, 25)

# Main UI
st.title("🩺 MediAssist — AI Disease Prediction")
st.caption("Enter symptoms to receive AI-based differentials. Click 'View Info' to get care tips for a disease.")

col1, col2 = st.columns([1.2, 1])
with col1:
    free_text = st.text_area("Type symptoms (comma-separated)", placeholder="e.g., fever, cough, headache")
    with st.expander("Or select symptoms (recommended)"):
        selected_tags = st.multiselect("Select symptoms:", options=sorted(set(ALIAS_MAP.values())))
    if st.button("🔍 Predict Disease", use_container_width=True):
        st.session_state["_run_pred"] = True
        st.session_state["selected_disease_norm"] = ""

with col2:
    st.markdown("**Tips**")
    st.markdown("- Use the selector for best accuracy.\n- Free text like 'fever and cough' is supported.\n- Click 'View Info' next to a predicted disease to see precautions.")

# Ensure session state keys
if "selected_disease_norm" not in st.session_state:
    st.session_state["selected_disease_norm"] = ""
if "disease_search" not in st.session_state:
    st.session_state["disease_search"] = ""

# ----------------------
# Prediction logic
# ----------------------
if st.session_state.get("_run_pred"):
    tokens = parse_free_text(free_text)
    recognized = []
    unmatched = []

    for tag in selected_tags:
        if tag in FEATURES and tag not in recognized:
            recognized.append(tag)

    for tok in tokens:
        can, matched_key, score = match_symptom(tok, ALIAS_MAP, threshold=fuzzy_threshold)
        if can and can not in recognized:
            recognized.append(can)
        else:
            unmatched.append(tok)

    st.markdown("---")
    st.markdown("#### ✅ Recognized Symptoms")
    if recognized:
        st.success(", ".join(sorted(recognized)))
    else:
        st.info("No valid symptoms recognized. Try the selector or reword your input.")

    if unmatched:
        st.warning("Auto-corrected / unrecognized inputs: " + ", ".join(unmatched))

    if len(recognized) < min_symptoms_needed:
        st.error(f"Add at least {min_symptoms_needed} symptom(s) to predict.")
        st.stop()

    X = symptoms_to_vector(recognized, FEATURES)
    try:
        proba = model.predict_proba([X])[0]
        raw_classes = list(model.classes_)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # map class labels to normalized disease keys
    def map_label_to_norm(label):
        try:
            if isinstance(label, bytes):
                label = label.decode()
        except Exception:
            pass
        s = str(label)
        ns = norm(s)
        if ns in norm2orig:
            return ns
        # try numeric index
        try:
            idx = int(float(s))
            if 0 <= idx < len(DISEASE_NORM_LIST):
                return DISEASE_NORM_LIST[idx]
        except Exception:
            pass
        # substring match
        for k in DISEASE_NORM_LIST:
            if ns in k or k in ns:
                return k
        # fuzzy fallback
        if fuzz_process is not None and DISEASE_NORM_LIST:
            best = fuzz_process.extractOne(ns, DISEASE_NORM_LIST)
            if best and best[1] >= 60:
                return best[0]
        return ns

    mapped_norms = [map_label_to_norm(c) for c in raw_classes]
    mapped_names = [norm2orig.get(k, str(raw_classes[i])) for i, k in enumerate(mapped_norms)]

    k = min(3, len(mapped_names))
    top_idx = np.argsort(proba)[-k:][::-1]
    top = [(mapped_norms[i], mapped_names[i], float(proba[i])) for i in top_idx]

    if top and (top[0][2] * 100) < min_prob_warn:
        st.warning(f"Low confidence (top {top[0][2]*100:.1f}%). Consider adding more symptoms.")

    st.subheader("🤖 AI Predictions (Top 3)")
    for rank, (norm_key, display_name, p) in enumerate(top, start=1):
        left, right = st.columns([3, 1])
        left.markdown(f"**{rank}. {display_name}** — Confidence: `{p*100:.2f}%`")
        left.progress(min(max(int(p * 100), 0), 100))
        if right.button("View Info", key=f"view_{rank}"):
            st.session_state["selected_disease_norm"] = norm_key
            st.session_state["disease_search"] = norm2orig.get(norm_key, display_name)

    st.info("ℹ️ Predictions are probabilistic. Use the 'View Info' button or the search box below to get care tips.")

# ----------------------
# Disease info display (clean bullets, no indices)
# ----------------------
# Use either the selected (clicked) disease or typed search value
selected_norm = st.session_state.get("selected_disease_norm")
# disease search input (prefilled if a predicted disease was clicked)
disease_search = st.text_input("Type a disease name:", value=st.session_state.get("disease_search", ""))
if disease_search and not selected_norm:
    # try to resolve typed name to a normalized key
    qk = None
    nq = norm(disease_search)
    if nq in norm2orig:
        qk = nq
    else:
        # substring or fuzzy
        for k in DISEASE_NORM_LIST:
            if nq in k or k in nq:
                qk = k
                break
        if qk is None and fuzz_process is not None and DISEASE_NORM_LIST:
            best = fuzz_process.extractOne(nq, DISEASE_NORM_LIST)
            if best and best[1] >= 60:
                qk = best[0]
    if qk:
        selected_norm = qk
        st.session_state["selected_disease_norm"] = qk
        st.session_state["disease_search"] = norm2orig.get(qk, disease_search)

if selected_norm:
    sel = norm2orig.get(selected_norm, None) or st.session_state.get("disease_search")
    st.markdown("---")
    st.subheader(f"🩺 Disease: {sel}")

    # description
    desc = disease_description.get(selected_norm, "No description available.")
    if desc:
        st.write(desc)

    # Precautions
    prec = disease_precautions.get(selected_norm, [])
    if prec:
        with st.expander("🔐 Precautions"):
            for p in prec:
                st.markdown(f"- {p}")

    # Medications
    meds = disease_medications.get(selected_norm, [])
    if meds:
        with st.expander("💊 Medications"):
            for m in meds:
                st.markdown(f"- {m}")

    # Diet
    diet = disease_diets.get(selected_norm, [])
    if diet:
        with st.expander("🥗 Recommended Diet"):
            for d in diet:
                st.markdown(f"- {d}")

    # Exercises
    work = disease_exercises.get(selected_norm, [])
    if work:
        with st.expander("🏃 Recommended Exercises"):
            for w in work:
                st.markdown(f"- {w}")

    st.info("ℹ️ This information is for educational purposes only. Consult a healthcare professional for diagnosis and treatment.")

# Footer note
st.markdown("\n---\n*App by MediAssist — not a substitute for professional medical advice.*")
