import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib
import os

# Load dataset
df = pd.read_csv('heart.csv')

# Load trained model safely
model_path = 'classification_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Error: Model file not found. Please check 'classification_model.pkl'.")
    model = None

# Validate model
if model is None:
    raise ValueError("The loaded model is None. Check the file path or format.")

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', StandardScaler(), ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                   'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    ]
)

# Create a pipeline with the classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Fit the pipeline only if the model is valid
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']
try:
    pipeline.fit(X, y)
except Exception as e:
    st.error(f"Pipeline fitting error: {e}")

# Prediction function
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    prediction = pipeline.predict(input_data)[0]
    return "\U0001F534 High Risk: Heart Disease Detected" if prediction == 1 else "\U0001F7E2 Low Risk: No Heart Disease"

# Streamlit App
def main():
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>❤️ Heart Disease Prediction ❤️</h1>
        <p style='text-align: center;'>Enter your health details in the sidebar to check your risk level.</p>
    """, unsafe_allow_html=True)

    # Sidebar for user inputs
    st.sidebar.header("🔍 Enter Your Health Details")
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, step=1)
    sex = st.sidebar.selectbox('Sex', ["Female (0)", "Male (1)"])
    cp = st.sidebar.selectbox('Chest Pain Type', ["0 - Typical Angina", "1 - Atypical Angina", "2 - Non-Anginal", "3 - Asymptomatic"])
    trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, step=1)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ["No (0)", "Yes (1)"])
    restecg = st.sidebar.selectbox('Resting ECG', ["0 - Normal", "1 - ST-T Wave Abnormality", "2 - LV Hypertrophy"])
    thalach = st.sidebar.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, step=1)
    exang = st.sidebar.selectbox('Exercise-Induced Angina', ["No (0)", "Yes (1)"])
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, step=0.1)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', ["0 - Upsloping", "1 - Flat", "2 - Downsloping"])
    ca = st.sidebar.selectbox('Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thalassemia', ["1 - Normal", "2 - Fixed defect", "3 - Reversible defect"])
    
    # Convert categorical inputs to numeric
    sex = 1 if sex == "Male (1)" else 0
    cp = int(cp[0])
    fbs = 1 if fbs == "Yes (1)" else 0
    restecg = int(restecg[0])
    exang = 1 if exang == "Yes (1)" else 0
    slope = int(slope[0])
    thal = int(thal[0])

    if st.sidebar.button('🔎 Predict'):
        prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        st.markdown(f"""
            <div style='text-align: center; font-size: 24px; font-weight: bold; padding: 15px; 
                border-radius: 10px; color: white; background-color: {'#FF4B4B' if 'High' in prediction else '#4CAF50'};'>
                {prediction}
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
