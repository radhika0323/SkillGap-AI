# 🎯 SkillGap AI

AI-powered Job Readiness & Skill Gap Predictor.

SkillGap AI analyzes a candidate's skills, projects, assessment score, and target role to estimate job readiness and identify important skill gaps.

## 🚀 Features

- Job readiness prediction using Machine Learning
- Target role selection
- Candidate skill assessment
- Skill gap identification
- Personalized improvement suggestions
- Interactive Streamlit interface
- Random Forest classification model

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## 📊 Input Parameters

The application considers:

- Python
- SQL
- Machine Learning
- Data Analysis
- Statistics
- Communication
- Number of Projects
- Assessment Score
- Target Role

## 🤖 Machine Learning

A Random Forest Classifier is used to predict whether a candidate is job-ready based on the available candidate profile features.

The trained model is saved using Joblib and loaded by the Streamlit application.

## 🖥️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/radhika0323/SkillGap-AI.git
cd SkillGap-AI
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app/app.py
```

## 📁 Project Structure

```text
SkillGap-AI/
│
├── app/
│   └── app.py
│
├── data/
│   └── skillgap_data.csv
│
├── models/
│   └── skillgap_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   └── train_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🎯 Future Improvements

- Use a larger real-world/public dataset
- Add more job roles
- Add explainable AI features
- Add personalized learning paths
- Improve model evaluation with larger datasets
- Deploy the application online

## 👩‍💻 Author

**Pobbathi Radhika**

B.Tech CSE (AI & ML)

GitHub: https://github.com/radhika0323