import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vidya Predictor", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #f9f9f9;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #FF9933;
        font-family: 'Inter', sans-serif;
    }
    h2, h3 {
        color: #138808;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #138808;
    }
    .stButton>button {
        background-color: #FF9933;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/students.csv")

@st.cache_resource
def load_models():
    best_model = None
    reg_model = None
    if os.path.exists('models/best_model.pkl'):
        best_model = joblib.load('models/best_model.pkl')
    if os.path.exists('models/reg_model.pkl'):
        reg_model = joblib.load('models/reg_model.pkl')
    return best_model, reg_model

@st.cache_data
def load_cv_results():
    if os.path.exists('models/cv_results.json'):
        with open('models/cv_results.json', 'r') as f:
            return json.load(f)
    return None

df = load_data()
best_model, reg_model = load_models()
cv_results = load_cv_results()

st.sidebar.title("🎓 Vidya Predictor")
st.sidebar.markdown("**Student Performance AI**\n\n*Empowering Indian educators with data-driven insights*")
page = st.sidebar.radio("Go to", ["Predict", "Model Comparison", "EDA Dashboard", "At-Risk Students"])

if page == "Predict":
    st.title("🎓 Vidya Predictor — Student Performance AI")
    st.markdown("Predict the final academic performance and risk level of a student based on their profile.")
    if best_model is None or reg_model is None:
        st.error("Models not found! Please run `python src/train.py` first.")
    else:
        with st.form("student_form"):
            st.subheader("Section A — Personal Details")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", "Ravi Kumar")
                gender = st.selectbox("Gender", ["Male", "Female"])
                state = st.selectbox("State", ['Maharashtra', 'UP', 'Tamil Nadu', 'Karnataka', 'Delhi', 'Bihar', 'Rajasthan', 'West Bengal', 'Gujarat', 'Telangana'])
                siblings = st.slider("Number of Siblings", 0, 5, 1)
            with col2:
                family_income = st.selectbox("Family Income", ['Low (< 2L)', 'Middle (2L-8L)', 'Upper Middle (8L-25L)', 'High (> 25L)'])
                parent_education = st.selectbox("Parent Education", ['Illiterate', 'Primary', 'Secondary', 'Graduate', 'Post-Graduate'])
                parent_occupation = st.selectbox("Parent Occupation", ['Farmer', 'Labour', 'Business', 'Government Job', 'Private Job', 'Professional'])
            st.subheader("Section B — Academic Background")
            col3, col4 = st.columns(2)
            with col3:
                board = st.selectbox("Board", ['CBSE', 'ICSE', 'State Board'])
                school_type = st.selectbox("School Type", ['Government', 'Private', 'Government-Aided'])
                medium = st.selectbox("Medium of Instruction", ['English', 'Hindi', 'Regional Language'])
            with col4:
                class_10_percent = st.slider("Class 10 Percentage", 40.0, 100.0, 75.0)
                class_12_percent = st.slider("Class 12 Percentage", 40.0, 100.0, 75.0)
                mid_term_marks = st.slider("Mid Term Marks (Out of 100)", 20, 100, 60)
                assignment_completion = st.slider("Assignment Completion %", 40.0, 100.0, 80.0)
            st.subheader("Section C — Study Habits & Resources")
            col5, col6 = st.columns(2)
            with col5:
                study_hours_per_day = st.slider("Daily Study Hours", 1.0, 12.0, 4.0)
                sleep_hours = st.slider("Sleep Hours per Night", 4.0, 9.0, 7.0)
                attendance_percent = st.slider("Attendance %", 50.0, 100.0, 85.0)
                stress_level = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
                hostel_or_day = st.radio("Residency", ['Day Scholar', 'Hostel'])
            with col6:
                tuition = st.selectbox("Takes Tuition", ["Yes", "No"])
                coaching_institute = st.selectbox("Attends Coaching Institute", ["Yes", "No"])
                internet_access = st.selectbox("Has Internet Access", ["Yes", "No"])
                smartphone_access = st.selectbox("Has Smartphone", ["Yes", "No"])
                extracurricular = st.selectbox("Participates in Extracurricular", ["Yes", "No"])
            submit_button = st.form_submit_button(label='Predict Performance')

        if submit_button:
            input_data = pd.DataFrame([{
                'gender': gender, 'state': state, 'board': board, 'school_type': school_type,
                'medium': medium, 'class_10_percent': class_10_percent, 'class_12_percent': class_12_percent,
                'attendance_percent': attendance_percent, 'study_hours_per_day': study_hours_per_day,
                'tuition': tuition, 'coaching_institute': coaching_institute, 'internet_access': internet_access,
                'smartphone_access': smartphone_access, 'family_income': family_income,
                'parent_education': parent_education, 'parent_occupation': parent_occupation,
                'siblings': siblings, 'hostel_or_day': hostel_or_day, 'extracurricular': extracurricular,
                'stress_level': stress_level, 'sleep_hours': sleep_hours, 'mid_term_marks': mid_term_marks,
                'assignment_completion': assignment_completion
            }])
            with st.spinner('Analyzing...'):
                prob_pass = best_model.predict_proba(input_data)[0][1]
                prob_fail = 1 - prob_pass
                pred_marks = reg_model.predict(input_data)[0]
                if pred_marks >= 90:
                    grade = "O (Outstanding)"
                elif pred_marks >= 75:
                    grade = "A+ (Excellent)"
                elif pred_marks >= 60:
                    grade = "A (Very Good)"
                elif pred_marks >= 50:
                    grade = "B+ (Good)"
                elif pred_marks >= 35:
                    grade = "B (Average — Pass)"
                else:
                    grade = "F (Fail)"
                if prob_fail > 0.7:
                    risk_badge = "🚨 High Risk — Needs Intervention"
                    msg = "मेहनत करो! You can do better!"
                    color = "red"
                elif prob_fail > 0.3:
                    risk_badge = "⚠️ Medium Risk"
                    msg = "ध्यान दो! Keep pushing yourself!"
                    color = "orange"
                else:
                    risk_badge = "✅ Low Risk"
                    msg = "शाबाश! Keep it up!"
                    color = "green"
                st.markdown("---")
                st.subheader(f"Prediction for {name}")
                st.markdown(f"<h3 style='text-align:center; color:{color};'>{msg}</h3>", unsafe_allow_html=True)
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown(f"<div class='metric-card'><h3>Predicted Final Marks</h3><h1 style='color:#FF9933;'>{pred_marks:.0f} / 100</h1></div>", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"<div class='metric-card'><h3>Expected Grade</h3><h1 style='color:#138808;'>{grade}</h1></div>", unsafe_allow_html=True)
                with rc3:
                    st.markdown(f"<div class='metric-card'><h3>Risk Status</h3><h2 style='color:{color};'>{risk_badge}</h2></div>", unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("Explainability (SHAP)")
                try:
                    preprocessor = best_model.named_steps['preprocessor']
                    classifier = best_model.named_steps['classifier']
                    X_transformed = preprocessor.transform(input_data)
                    numeric_cols = preprocessor.transformers_[0][2]
                    cat_cols = preprocessor.transformers_[1][2]
                    feature_names = numeric_cols + cat_cols
                    explainer = shap.Explainer(classifier)
                    shap_values = explainer(X_transformed)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if len(shap_values.values.shape) == 3:
                        shap.waterfall_plot(shap.Explanation(values=shap_values.values[0,:,1], 
                                                             base_values=shap_values.base_values[0,1], 
                                                             data=X_transformed[0], 
                                                             feature_names=feature_names), show=False)
                    else:
                        shap.waterfall_plot(shap.Explanation(values=shap_values.values[0], 
                                                             base_values=shap_values.base_values[0], 
                                                             data=X_transformed[0], 
                                                             feature_names=feature_names), show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning("SHAP explanation not fully supported for this model pipeline type.")

elif page == "Model Comparison":
    st.title("📊 Which AI model best predicts student success in India?")
    if cv_results is None:
        st.error("Evaluation results not found! Please run `python src/train.py` first.")
    else:
        results_df = pd.DataFrame(cv_results)
        results_df.set_index('Model', inplace=True)
        st.subheader("Performance Metrics (Cross-Validation)")
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
        st.markdown("---")
        st.subheader("F1-Score Comparison")
        fig = px.bar(results_df.reset_index(), x='Model', y='F1-Score', color='F1-Score', color_continuous_scale='YlOrRd', text='F1-Score')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

elif page == "EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Marks Distribution across Indian Students")
        fig_marks = px.histogram(df, x="final_marks", nbins=20, marginal="box", color_discrete_sequence=['#FF9933'])
        st.plotly_chart(fig_marks, use_container_width=True)
    with col2:
        st.subheader("Pass vs Fail Rate")
        fig_pie = px.pie(df, names='result', color='result', color_discrete_map={'Pass':'#138808', 'Fail':'#FF0000'}, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Performance by Board")
        fig_board = px.bar(df.groupby('board')['final_marks'].mean().reset_index(), x='board', y='final_marks', color='board')
        st.plotly_chart(fig_board, use_container_width=True)
    with col4:
        st.subheader("Impact of Coaching Institute on Final Marks")
        fig_coaching = px.box(df, x="coaching_institute", y="final_marks", color="coaching_institute")
        st.plotly_chart(fig_coaching, use_container_width=True)
    st.markdown("---")
    st.subheader("Study Hours vs Final Marks (colored by Family Income)")
    fig_scatter = px.scatter(df, x="study_hours_per_day", y="final_marks", color="family_income", opacity=0.7)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("---")
    st.subheader("State-wise Average Performance")
    state_perf = df.groupby('state')['final_marks'].mean().reset_index().sort_values('final_marks')
    fig_state = px.bar(state_perf, x="final_marks", y="state", orientation='h', color='final_marks', color_continuous_scale='Greens')
    st.plotly_chart(fig_state, use_container_width=True)
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "At-Risk Students":
    st.title("🚨 At-Risk Students Dashboard")
    if best_model is None or reg_model is None:
        st.error("Models not found! Please run `python src/train.py` first.")
    else:
        st.markdown("This dashboard identifies students who are at high risk of failing (< 35 marks) based on AI predictions.")
        with st.spinner("Analyzing all students..."):
            X_all = df.drop(columns=['final_marks', 'result', 'name'])
            probs = best_model.predict_proba(X_all)[:, 0]
            df_pred = df.copy()
            df_pred['prob_fail'] = probs
            df_pred['Predicted Marks'] = reg_model.predict(X_all)
            df_high_risk = df_pred[df_pred['prob_fail'] > 0.7].copy()
            df_high_risk['Risk Level'] = "High Risk"
        st.subheader("Filter High-Risk Students")
        col1, col2, col3, col4 = st.columns(4)
        state_filter = col1.selectbox("Filter by State", ["All"] + list(df['state'].unique()))
        board_filter = col2.selectbox("Filter by Board", ["All"] + list(df['board'].unique()))
        school_filter = col3.selectbox("Filter by School Type", ["All"] + list(df['school_type'].unique()))
        income_filter = col4.selectbox("Filter by Family Income", ["All"] + list(df['family_income'].unique()))
        filtered_df = df_high_risk.copy()
        if state_filter != "All": filtered_df = filtered_df[filtered_df['state'] == state_filter]
        if board_filter != "All": filtered_df = filtered_df[filtered_df['board'] == board_filter]
        if school_filter != "All": filtered_df = filtered_df[filtered_df['school_type'] == school_filter]
        if income_filter != "All": filtered_df = filtered_df[filtered_df['family_income'] == income_filter]
        display_df = filtered_df[['name', 'state', 'board', 'Predicted Marks', 'Risk Level']]
        st.markdown("### Summary Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total At-Risk Students", len(filtered_df))
        if len(filtered_df) > 0:
            c2.metric("Most Affected State", filtered_df['state'].mode()[0])
            c3.metric("Most Affected Board", filtered_df['board'].mode()[0])
            st.dataframe(display_df, use_container_width=True)
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download as CSV", data=csv, file_name="high_risk_students.csv", mime="text/csv")
        else:
            st.success("No high-risk students found matching the criteria!")