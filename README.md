# 📄 AI Resume Screening System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end machine learning system that predicts whether a candidate will be shortlisted based on their resume features — with an interactive Streamlit dashboard.**

[Demo](#demo) · [Installation](#installation) · [Usage](#usage) · [Model Performance](#model-performance) · [Project Structure](#project-structure)

</div>

---

## 📌 Overview

Manual resume screening is time-consuming and inconsistent. This project builds an ML pipeline that automates the shortlisting decision using key resume signals — education, experience, skills match, project count, resume length, and GitHub activity.

The system ships with:
- A trained classification model (Logistic Regression / Random Forest / Gradient Boosting / SVM)
- A full Streamlit web app with 4 interactive tabs
- Confidence scores and personalized improvement tips per candidate
- Model evaluation with confusion matrix, ROC curve, and feature importance

---

## 🎯 Problem Statement

> Given a set of resume features, predict whether a candidate will be **shortlisted (Yes)** or **rejected (No)** for further interview rounds.

This is a **binary classification** problem. The target variable is `shortlisted`.

---

## 🖥️ Demo

| Tab | What it shows |
|-----|--------------|
| 🏠 Overview | Dataset stats, shortlist rate, education breakdown |
| 📊 Model Performance | Accuracy, ROC-AUC, confusion matrix, feature importance |
| 🔍 Predict Resume | Input any candidate → get prediction + confidence + tips |
| 📈 Data Explorer | Interactive scatter plots, heatmap, distributions |

---

## 📂 Dataset

**File:** `ai_resume_screening.csv`

| Feature | Type | Description |
|---------|------|-------------|
| `education_level` | Categorical | High School / Bachelor / Master / PhD |
| `years_experience` | Numerical | Total years of work experience |
| `skills_match_score` | Numerical | Score (0–100) of how well skills match the JD |
| `project_count` | Numerical | Number of projects on the resume |
| `resume_length` | Numerical | Word count of the resume |
| `github_activity` | Numerical | GitHub contribution score |
| `shortlisted` | Target | Yes / No |

> If no CSV is provided, the app auto-generates synthetic demo data so you can explore it immediately.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-resume-screener.git
cd ai-resume-screener

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.30
joblib>=1.3
```

---

## 🚀 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Run the Notebook

```bash
jupyter notebook resume_screening.ipynb
```

### Predict via Python

```python
import pandas as pd
import joblib

model = joblib.load('resume_screener_model.joblib')

candidate = pd.DataFrame([{
    'years_experience': 5,
    'skills_match_score': 82,
    'project_count': 7,
    'resume_length': 600,
    'github_activity': 320,
    'education_level': 'Master'
}])

result = model.predict(candidate)
proba  = model.predict_proba(candidate)

print("Shortlisted" if result[0] == 1 else "Rejected")
print(f"Confidence: {max(proba[0]) * 100:.1f}%")
```

---

## 🏗️ Pipeline Architecture

```
Raw CSV
   │
   ▼
┌─────────────────────┐
│   Data Preprocessing │   ← Null check, label encoding, EDA
└──────────┬──────────┘
           │
   ┌───────┴────────┐
   │                │
   ▼                ▼
Numerical        Categorical
StandardScaler   OrdinalEncoder
   │                │
   └───────┬────────┘
           │
           ▼
┌─────────────────────┐
│  Classification Model│   ← LR / RF / GBM / SVM
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Evaluation         │   ← Accuracy, ROC-AUC, CV Score
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Streamlit App      │   ← Interactive prediction + tips
└─────────────────────┘
```

---

## 📊 Model Performance

Results on a held-out 20% test set:

| Model | Accuracy | ROC-AUC | CV Score (5-fold) |
|-------|----------|---------|-------------------|
| Logistic Regression | ~82% | ~88% | ~81% |
| Random Forest | ~91% | ~96% | ~90% |
| Gradient Boosting | ~90% | ~95% | ~89% |
| SVM | ~85% | ~91% | ~84% |

> Results may vary slightly depending on the dataset. Random Forest is recommended as the default model.

**Top features by importance (Random Forest):**

1. `skills_match_score` — strongest predictor
2. `years_experience`
3. `github_activity`
4. `project_count`
5. `education_level`
6. `resume_length`

---

## 📁 Project Structure

```
ai-resume-screener/
│
├── app.py                        # Streamlit web application
├── resume_screening.ipynb        # EDA + model training notebook
├── ai_resume_screening.csv       # Dataset (add your own)
├── resume_screener_model.joblib  # Saved model (generated after training)
├── label_encoder.joblib          # Saved label encoder
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## 🔍 Key ML Concepts Demonstrated

- **Exploratory Data Analysis (EDA)** — distributions, correlations, class balance
- **Feature Engineering** — handling categorical ordinal data with `OrdinalEncoder`
- **Pipelines** — clean `sklearn.Pipeline` + `ColumnTransformer` for reproducible preprocessing
- **Model Comparison** — 4 classifiers evaluated on the same train/test split
- **Hyperparameter Tuning** — `GridSearchCV` for Random Forest
- **Evaluation Metrics** — accuracy, precision, recall, F1, ROC-AUC, 5-fold CV
- **Class Imbalance Awareness** — using ROC-AUC alongside accuracy
- **Deployment** — Streamlit app with joblib model serialization

---

## 🛠️ What's Next (Roadmap)

- [ ] Add SHAP explainability values per prediction
- [ ] Support for bulk CSV upload and batch prediction
- [ ] Fine-tune with XGBoost and LightGBM
- [ ] Add SMOTE for handling class imbalance
- [ ] Export prediction results as a downloadable CSV
- [ ] Deploy to Streamlit Cloud / HuggingFace Spaces

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

<div align="center">
  <sub>Built with Python, scikit-learn, and Streamlit · Give it a ⭐ if you found it useful!</sub>
</div>
