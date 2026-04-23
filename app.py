import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import  pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e9ecef;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.1;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #6c757d;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-shortlisted {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-rejected {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-sub {
        font-size: 1rem;
        color: #555;
    }

    .sidebar-section {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .sidebar-section h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #444;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    div[data-testid="stSelectbox"] > label,
    div[data-testid="stSlider"] > label,
    div[data-testid="stNumberInput"] > label {
        font-weight: 500;
        color: #333;
    }

    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #16213e;
        transform: translateY(-1px);
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #4f46e5;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    .tip-box {
        background: #eef2ff;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #4f46e5;
        margin-top: 16px;
        font-size: 0.88rem;
        color: #3730a3;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "resume_screener_model.joblib"
LABEL_PATH = "label_encoder.joblib"

NUM_COLS = ['years_experience', 'skills_match_score', 'project_count',
            'resume_length', 'github_activity']
CAT_COLS = ['education_level']

EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD']

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

# ── Helper: build pipeline ─────────────────────────────────────────────────────
def build_pipeline(clf):
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([
        ('ordinal', OrdinalEncoder(
            categories=[EDUCATION_LEVELS],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, NUM_COLS),
        ('cat', cat_pipe, CAT_COLS),
    ])
    return Pipeline([('preprocessor', pre), ('model', clf)])


# ── Helper: train and cache ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(df_hash, model_name, tune_hp):
    df = st.session_state['df'].copy()

    le = LabelEncoder()
    df['shortlisted'] = le.fit_transform(df['shortlisted'])

    X = df.drop(columns='shortlisted')
    y = df['shortlisted']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = MODELS[model_name]
    pipe = build_pipeline(clf)

    if tune_hp and model_name == "Random Forest":
        param_grid = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 5, 10],
        }
        gs = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        pipe = gs.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred) * 100
    roc     = roc_auc_score(y_test, y_proba) * 100
    cv      = cross_val_score(pipe, X, y, cv=5).mean() * 100
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=['Rejected', 'Shortlisted'],
                                    output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(le,   LABEL_PATH)

    feat_imp = None
    if model_name in ("Random Forest", "Gradient Boosting"):
        inner = pipe.named_steps['model']
        feat_imp = dict(zip(NUM_COLS + CAT_COLS, inner.feature_importances_))

    return {
        'pipe': pipe, 'le': le,
        'acc': acc, 'roc': roc, 'cv': cv,
        'cm': cm, 'report': report,
        'fpr': fpr, 'tpr': tpr,
        'feat_imp': feat_imp,
        'X_test': X_test, 'y_test': y_test,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 AI Resume Screener")
    st.markdown("---")

    # Upload
    st.markdown('<div class="sidebar-section"><h4>Dataset</h4>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload ai_resume_screening.csv", type="csv")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.session_state['df'] = df_raw
        st.success(f"Loaded {len(df_raw):,} rows")
    elif 'df' not in st.session_state:
        # Generate synthetic demo data if no file
        np.random.seed(42)
        n = 500
        edu = np.random.choice(['High School','Bachelor','Master','PhD'], n,
                               p=[0.15, 0.45, 0.30, 0.10])
        exp = np.random.randint(0, 20, n)
        skill = np.random.randint(40, 100, n)
        proj = np.random.randint(0, 15, n)
        res_len = np.random.randint(200, 900, n)
        github = np.random.randint(0, 600, n)
        edu_score = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        score = (np.vectorize(edu_score.get)(edu)*10 + exp*2 +
                 skill*0.5 + proj*3 + github*0.05)
        threshold = np.percentile(score, 45)
        shortlisted = np.where(score >= threshold, 'Yes', 'No')
        df_raw = pd.DataFrame({
            'education_level': edu, 'years_experience': exp,
            'skills_match_score': skill, 'project_count': proj,
            'resume_length': res_len, 'github_activity': github,
            'shortlisted': shortlisted,
        })
        st.session_state['df'] = df_raw
        st.info("Using synthetic demo data")

    st.markdown("---")

    # Model selection
    st.markdown('<div class="sidebar-section"><h4>Model Settings</h4>', unsafe_allow_html=True)
    chosen_model = st.selectbox("Algorithm", list(MODELS.keys()), index=1)
    tune_hp = st.checkbox("Hyperparameter tuning (Random Forest)", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    train_btn = st.button("Train Model")

    st.markdown("---")
    st.markdown("**Features used:**")
    for f in NUM_COLS + CAT_COLS:
        st.markdown(f"- `{f}`")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Overview", "📊 Model Performance", "🔍 Predict Resume", "📈 Data Explorer"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## AI Resume Screening System")
    st.markdown(
        "Upload your dataset and train a model to predict whether a candidate "
        "will be **shortlisted** based on their resume features.")

    df = st.session_state['df']

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    shortlisted_pct = (df['shortlisted'] == 'Yes').mean() * 100

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Candidates</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{shortlisted_pct:.1f}%</div>
            <div class="metric-label">Shortlisted Rate</div></div>""",
            unsafe_allow_html=True)
    with col3:
        avg_exp = df['years_experience'].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_exp:.1f}</div>
            <div class="metric-label">Avg Years Exp</div></div>""",
            unsafe_allow_html=True)
    with col4:
        avg_skill = df['skills_match_score'].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_skill:.1f}</div>
            <div class="metric-label">Avg Skill Score</div></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df['shortlisted'].value_counts()
        colors = ['#28a745', '#dc3545']
        ax.pie(counts, labels=['Shortlisted','Rejected'], autopct='%1.1f%%',
               colors=colors, startangle=140,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Shortlist Distribution', fontweight='bold', fontsize=13)
        fig.patch.set_facecolor('white')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        edu_short = df.groupby('education_level')['shortlisted'].apply(
            lambda x: (x == 'Yes').mean() * 100).reindex(EDUCATION_LEVELS)
        bars = ax.bar(edu_short.index, edu_short.values,
                      color=['#6c757d','#4f46e5','#7c3aed','#a855f7'],
                      edgecolor='white', linewidth=1.5)
        ax.set_xlabel('Education Level', fontsize=11)
        ax.set_ylabel('Shortlisted %', fontsize=11)
        ax.set_title('Shortlist Rate by Education', fontweight='bold', fontsize=13)
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, edu_short.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('white')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if not train_btn:
        st.markdown("""<div class="tip-box">
            👆 Select a model in the sidebar and click <strong>Train Model</strong> to see 
            performance metrics, feature importance, and start predicting resumes.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if train_btn or 'results' in st.session_state:
        if train_btn:
            df_hash = str(len(st.session_state['df'])) + chosen_model + str(tune_hp)
            with st.spinner(f"Training {chosen_model}..."):
                results = train_model(df_hash, chosen_model, tune_hp)
            st.session_state['results'] = results
            st.session_state['model_name'] = chosen_model

        results = st.session_state['results']

        st.markdown(f"## {st.session_state['model_name']} — Results")

        # Metrics row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#4f46e5">{results['acc']:.2f}%</div>
                <div class="metric-label">Test Accuracy</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#7c3aed">{results['roc']:.2f}%</div>
                <div class="metric-label">ROC-AUC Score</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#059669">{results['cv']:.2f}%</div>
                <div class="metric-label">CV Score (5-fold)</div></div>""",
                unsafe_allow_html=True)

        st.markdown('<div class="section-header">Detailed Metrics</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # Confusion matrix
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            cm = results['cm']
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Rejected', 'Shortlisted'])
            ax.set_yticklabels(['Rejected', 'Shortlisted'])
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            fontsize=16, fontweight='bold',
                            color='white' if cm[i,j] > cm.max()/2 else 'black')
            fig.patch.set_facecolor('white')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_right:
            # ROC curve
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.plot(results['fpr'], results['tpr'], color='#4f46e5',
                    linewidth=2.5, label=f"AUC = {results['roc']:.1f}%")
            ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Random')
            ax.fill_between(results['fpr'], results['tpr'], alpha=0.1, color='#4f46e5')
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title('ROC Curve', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            fig.patch.set_facecolor('white')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Classification report table
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        report = results['report']
        report_df = pd.DataFrame({
            'Class': ['Rejected', 'Shortlisted', 'Macro Avg', 'Weighted Avg'],
            'Precision': [f"{report['Rejected']['precision']:.3f}",
                          f"{report['Shortlisted']['precision']:.3f}",
                          f"{report['macro avg']['precision']:.3f}",
                          f"{report['weighted avg']['precision']:.3f}"],
            'Recall': [f"{report['Rejected']['recall']:.3f}",
                       f"{report['Shortlisted']['recall']:.3f}",
                       f"{report['macro avg']['recall']:.3f}",
                       f"{report['weighted avg']['recall']:.3f}"],
            'F1-Score': [f"{report['Rejected']['f1-score']:.3f}",
                         f"{report['Shortlisted']['f1-score']:.3f}",
                         f"{report['macro avg']['f1-score']:.3f}",
                         f"{report['weighted avg']['f1-score']:.3f}"],
            'Support': [report['Rejected']['support'],
                        report['Shortlisted']['support'],
                        report['macro avg']['support'],
                        report['weighted avg']['support']],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        # Feature importance
        if results['feat_imp']:
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            fi = dict(sorted(results['feat_imp'].items(), key=lambda x: x[1]))
            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors_fi = ['#4f46e5' if v == max(fi.values()) else '#a5b4fc' for v in fi.values()]
            bars = ax.barh(list(fi.keys()), list(fi.values()), color=colors_fi,
                           edgecolor='white', linewidth=1)
            ax.set_xlabel('Importance Score', fontsize=11)
            ax.set_title('Feature Importance', fontweight='bold', fontsize=13)
            for bar, val in zip(bars, fi.values()):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            fig.patch.set_facecolor('white')
            st.pyplot(fig, use_container_width=True)
            plt.close()
    else:
        st.info("Train a model first using the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Predict Resume
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Predict a Candidate's Outcome")

    if 'results' not in st.session_state:
        st.info("Please train a model first from the sidebar.")
    else:
        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("**Enter candidate details:**")

            edu = st.selectbox("Education Level", EDUCATION_LEVELS, index=2)
            exp = st.slider("Years of Experience", 0, 25, 5)
            skill = st.slider("Skills Match Score", 0, 100, 70)
            proj = st.slider("Project Count", 0, 20, 5)
            res_len = st.number_input("Resume Length (words)", 100, 1200, 550, step=50)
            github = st.number_input("GitHub Activity Score", 0, 700, 250, step=10)

            predict_btn = st.button("Predict Outcome")

        with col_result:
            if predict_btn:
                pipe = st.session_state['results']['pipe']
                features = NUM_COLS + CAT_COLS
                x_new = pd.DataFrame(
                    [[exp, skill, proj, res_len, github, edu]],
                    columns=features
                )

                pred = pipe.predict(x_new)[0]
                proba = pipe.predict_proba(x_new)[0]
                confidence = proba[pred] * 100

                if pred == 1:
                    st.markdown(f"""<div class="result-shortlisted">
                        <div class="result-title" style="color:#155724">✅ Shortlisted</div>
                        <div class="result-sub">Confidence: <strong>{confidence:.1f}%</strong></div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-rejected">
                        <div class="result-title" style="color:#721c24">❌ Rejected</div>
                        <div class="result-sub">Confidence: <strong>{confidence:.1f}%</strong></div>
                    </div>""", unsafe_allow_html=True)

                # Probability bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Prediction probabilities:**")
                prob_df = pd.DataFrame({
                    'Outcome': ['Rejected', 'Shortlisted'],
                    'Probability': [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    'Score': [proba[0], proba[1]]
                })

                fig, ax = plt.subplots(figsize=(5, 1.8))
                colors_p = ['#dc3545', '#28a745']
                ax.barh(['Rejected', 'Shortlisted'], [proba[0], proba[1]],
                        color=colors_p, height=0.5)
                for i, v in enumerate([proba[0], proba[1]]):
                    ax.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontsize=11, fontweight='bold')
                ax.set_xlim(0, 1.15)
                ax.spines[['top','right','left']].set_visible(False)
                ax.set_xticks([])
                fig.patch.set_facecolor('white')
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Resume tips
                st.markdown("**Improvement tips:**")
                tips = []
                if skill < 70:
                    tips.append("📚 Improve skills match score — target role-specific skills")
                if github < 200:
                    tips.append("💻 Increase GitHub activity — contribute to open source")
                if proj < 5:
                    tips.append("🛠️ Add more projects to demonstrate hands-on experience")
                if exp < 3:
                    tips.append("🕒 Consider internships or freelance work to gain experience")
                if edu == 'High School':
                    tips.append("🎓 Consider pursuing higher education for better chances")
                if not tips:
                    tips.append("✨ Strong profile — keep maintaining all these metrics!")

                for tip in tips:
                    st.markdown(f"- {tip}")
            else:
                st.markdown("""<div class="tip-box">
                    Fill in the candidate details on the left and click 
                    <strong>Predict Outcome</strong> to see the result with 
                    confidence score and personalized improvement tips.
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Explore the Dataset")
    df = st.session_state['df']

    col_x, col_y, col_hue = st.columns(3)
    num_options = NUM_COLS
    with col_x:
        x_feat = st.selectbox("X axis", num_options, index=0)
    with col_y:
        y_feat = st.selectbox("Y axis", num_options, index=1)
    with col_hue:
        hue_by = st.selectbox("Color by", ['shortlisted', 'education_level'])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = {'Yes': '#28a745', 'No': '#dc3545'}
    groups = df[hue_by].unique()
    cmap = plt.cm.get_cmap('Set2', len(groups))
    for i, grp in enumerate(groups):
        sub = df[df[hue_by] == grp]
        color = palette.get(grp, cmap(i))
        ax.scatter(sub[x_feat], sub[y_feat], label=grp, alpha=0.6,
                   color=color, s=30, edgecolors='none')
    ax.set_xlabel(x_feat.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y_feat.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f'{x_feat} vs {y_feat} by {hue_by}', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    df_num = df[NUM_COLS].copy()
    corr = df_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    import seaborn as sns
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, mask=mask, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
    fig.patch.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">Distribution by Shortlisted Status</div>', unsafe_allow_html=True)
    sel_feat = st.selectbox("Select feature to compare", NUM_COLS)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for outcome, color in [('Yes', '#28a745'), ('No', '#dc3545')]:
        sub = df[df['shortlisted'] == outcome][sel_feat]
        ax.hist(sub, bins=25, alpha=0.6, color=color, label=outcome, edgecolor='white')
    ax.set_xlabel(sel_feat.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{sel_feat} distribution by shortlist status', fontweight='bold', fontsize=13)
    ax.legend(['Shortlisted', 'Rejected'], fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close()
