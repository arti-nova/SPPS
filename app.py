import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — premium look
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef1f8 100%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 50%, #9333ea 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0;
    }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border: 1px solid #eef0f5;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid #eef0f5;
        text-align: center;
    }
    .pass-badge {
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }
    .pass-good {
        background: #dcfce7;
        color: #15803d;
    }
    .pass-bad {
        background: #fee2e2;
        color: #b91c1c;
    }

    /* Buttons */
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #4338ca, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 14px rgba(76, 29, 149, 0.3);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 29, 149, 0.4);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1e1b4b;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }

    /* Metric label tweak */
    div[data-testid="stMetricValue"] {
        font-size: 2.4rem;
        font-weight: 800;
        color: #4338ca;
    }

    footer, #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_rf_no_grades.pkl")
    scaler = joblib.load("models/scaler_no_grades.pkl")
    feature_columns = joblib.load("models/feature_columns_no_grades.pkl")
    numeric_cols = joblib.load("models/numeric_cols_no_grades.pkl")
    return model, scaler, feature_columns, numeric_cols

try:
    model, scaler, feature_columns, numeric_cols = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Predicts final grade (G3, 0–20 scale) from attendance, behavior, and demographics —
    no prior grades required. Powered by a tuned Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error("Model artifacts not found in `models/`. Run the export cell in `modeling.ipynb` first.")
    st.stop()

# ============================================================
# SIDEBAR — quick info
# ============================================================
with st.sidebar:
    st.markdown("### 📊 About this model")
    st.markdown("""
    - **Algorithm:** Random Forest (tuned via GridSearchCV)
    - **Features used:** Demographics, family background,
      study habits, attendance — *no G1/G2 prior grades*
    - **Target:** Final grade G3 (0–20 scale)
    - **Dataset:** UCI Student Performance
    """)
    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown("Fill in the student's details in each section, then click **Predict Final Grade**.")

# ============================================================
# INPUT FORM
# ============================================================
with st.form("student_form"):

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧑 Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
    with col2:
        address = st.selectbox("Address type", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"], format_func=lambda x: "≤3" if x == "LE3" else ">3")
        Pstatus = st.selectbox("Parent status", ["T", "A"], format_func=lambda x: "Together" if x == "T" else "Apart")
    with col3:
        Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
        Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👨‍👩‍👧 Family & Support</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
    with col2:
        schoolsup = st.checkbox("Extra school support")
        famsup = st.checkbox("Family educational support")
        paid = st.checkbox("Extra paid classes")
        activities = st.checkbox("Extracurricular activities")
        nursery = st.checkbox("Attended nursery school")
        higher = st.checkbox("Wants higher education", value=True)
        internet = st.checkbox("Internet access at home", value=True)
        romantic = st.checkbox("In a romantic relationship")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📚 Study & Behavior</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        traveltime = st.slider("Travel time to school (1=short, 4=long)", 1, 4, 1)
        studytime = st.slider("Weekly study time (1=low, 4=high)", 1, 4, 2)
        failures = st.slider("Past class failures", 0, 4, 0)
        absences = st.slider("Number of absences", 0, 75, 4)
    with col2:
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        health = st.slider("Current health status (1-5)", 1, 5, 4)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🍷 Alcohol Consumption</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        Dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
    with col2:
        Walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 1)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮 Predict Final Grade")

# ============================================================
# PREDICTION
# ============================================================
if submitted:
    raw_input = {
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": "yes" if schoolsup else "no",
        "famsup": "yes" if famsup else "no",
        "paid": "yes" if paid else "no",
        "activities": "yes" if activities else "no",
        "nursery": "yes" if nursery else "no",
        "higher": "yes" if higher else "no",
        "internet": "yes" if internet else "no",
        "romantic": "yes" if romantic else "no",
        "famrel": famrel, "freetime": freetime, "goout": goout,
        "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences,
    }
    input_df = pd.DataFrame([raw_input])

    binary_cols = ["schoolsup", "famsup", "paid", "activities",
                    "nursery", "higher", "internet", "romantic"]
    for col in binary_cols:
        input_df[col] = input_df[col].map({"yes": 1, "no": 0})

    input_df["school"] = input_df["school"].map({"GP": 0, "MS": 1})
    input_df["sex"] = input_df["sex"].map({"F": 0, "M": 1})
    input_df["address"] = input_df["address"].map({"U": 0, "R": 1})
    input_df["famsize"] = input_df["famsize"].map({"LE3": 0, "GT3": 1})
    input_df["Pstatus"] = input_df["Pstatus"].map({"T": 0, "A": 1})

    input_df = pd.get_dummies(input_df, columns=["Mjob", "Fjob", "reason", "guardian"], drop_first=True)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = float(np.clip(model.predict(input_df)[0], 0, 20))
    is_passing = prediction >= 10

    st.markdown("<br>", unsafe_allow_html=True)
    result_col, gauge_col = st.columns([1, 1.4])

    with result_col:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### Predicted Final Grade")
        st.metric(label="", value=f"{prediction:.1f} / 20")
        badge_class = "pass-good" if is_passing else "pass-bad"
        badge_text = "✅ Passing" if is_passing else "⚠️ Below Passing"
        st.markdown(f'<span class="pass-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        if is_passing:
            st.success("This student's predicted grade is at or above the passing threshold (10/20).")
        else:
            st.warning("This student's predicted grade is below the passing threshold — consider additional academic support.")
        st.markdown('</div>', unsafe_allow_html=True)

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'suffix': " / 20", 'font': {'size': 40, 'color': '#4338ca'}},
            gauge={
                'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#4338ca", 'thickness': 0.35},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 10], 'color': '#fee2e2'},
                    {'range': [10, 14], 'color': '#fef9c3'},
                    {'range': [14, 20], 'color': '#dcfce7'},
                ],
                'threshold': {
                    'line': {'color': "#1e1b4b", 'width': 4},
                    'thickness': 0.85,
                    'value': prediction
                }
            }
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'family': "Inter, sans-serif"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View encoded model input (for debugging)"):
        st.dataframe(input_df, use_container_width=True)