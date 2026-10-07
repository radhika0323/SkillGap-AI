import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SkillGap AI",
    page_icon="🎯",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🎯 SkillGap AI")
st.write("AI-powered Job Readiness & Skill Gap Predictor")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/skillgap_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# CANDIDATE PROFILE
# -----------------------------
st.subheader("👤 Candidate Profile")

assessment_score = st.slider(
    "Assessment Score",
    min_value=0,
    max_value=100,
    value=60
)

projects = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=10,
    value=2
)

target_role = st.selectbox(
    "Target Role",
    ["AI_ML", "Data_Analyst"]
)

st.subheader("🧑‍💻 Candidate Skills")

python_skill = st.checkbox("Python")
sql_skill = st.checkbox("SQL")
machine_learning = st.checkbox("Machine Learning")
communication = st.checkbox("Communication")
statistics = st.checkbox("Statistics")
data_analysis = st.checkbox("Data Analysis")

# -----------------------------
# TARGET ROLE ENCODING
# -----------------------------
if target_role == "AI_ML":
    target_role_value = 1
else:
    target_role_value = 0

# -----------------------------
# INPUT DATA
# -----------------------------
input_data = pd.DataFrame([{
    "assessment_score": assessment_score,
    "communication": int(communication),
    "data_analysis": int(data_analysis),
    "machine_learning": int(machine_learning),
    "projects": projects,
    "python": int(python_skill),
    "sql": int(sql_skill),
    "statistics": int(statistics),
    "target_role": target_role_value
}])

# -----------------------------
# SHOW PROFILE
# -----------------------------
st.subheader("📊 Candidate Profile")

st.dataframe(input_data, use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Check Job Readiness", use_container_width=True):

    try:

        # IMPORTANT:
        # Match the exact feature order used during training
        if hasattr(model, "feature_names_in_"):

            expected_features = list(model.feature_names_in_)

            input_data = input_data.reindex(
                columns=expected_features,
                fill_value=0
            )

        prediction = model.predict(input_data)[0]

        # -----------------------------
        # READINESS SCORE
        # -----------------------------
        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)[0]

            readiness = max(probability) * 100

        else:

            readiness = 70 if prediction == 1 else 40

        # -----------------------------
        # RESULT
        # -----------------------------
        st.subheader("🎯 Job Readiness Result")

        if prediction == 1:

            st.success(
                "🎉 You are Job Ready!"
            )

        else:

            st.warning(
                "📚 You need more preparation."
            )

        st.metric(
            "Job Readiness Score",
            f"{readiness:.1f}%"
        )

        # -----------------------------
        # SKILL GAP
        # -----------------------------
        st.subheader("💡 Skill Gap Suggestions")

        suggestions = []

        if not python_skill:
            suggestions.append("Improve Python")

        if not sql_skill:
            suggestions.append("Learn SQL")

        if not machine_learning:
            suggestions.append("Learn Machine Learning")

        if not communication:
            suggestions.append("Improve Communication")

        if not statistics:
            suggestions.append("Strengthen Statistics")

        if not data_analysis:
            suggestions.append("Improve Data Analysis")

        if projects < 2:
            suggestions.append("Build more practical projects")

        if assessment_score < 60:
            suggestions.append("Improve technical assessment performance")

        if suggestions:

            for item in suggestions:
                st.write("• " + item)

        else:

            st.success(
                "🔥 Excellent! You have covered the major skills."
            )

    except Exception as e:

        st.error("❌ Prediction Error")

        st.code(str(e))

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "SkillGap AI | AI/ML Career Readiness & Skill Gap Predictor"
)