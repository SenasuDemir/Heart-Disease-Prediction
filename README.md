# ❤️ Heart Disease Prediction Model 🚑

## Introduction
Cardiovascular diseases are one of the leading causes of mortality worldwide. Early diagnosis of heart disease can significantly improve treatment outcomes and reduce health risks. This project aims to develop a machine learning model that predicts the likelihood of a person having heart disease based on various medical attributes, providing a valuable tool for healthcare professionals. 🩺

## Aim 🎯
The goal of this project is to build a machine learning model that predicts the likelihood of a person having heart disease based on their medical attributes. By analyzing key health indicators, the model can assist in early diagnosis and support clinical decision-making.

## Dataset Description 📊
The dataset consists of multiple features that provide information about the patient's health condition. Below are the descriptions of each column:

- **age**: Age of the patient in years.
- **sex**: Gender of the patient (1 = male, 0 = female).
- **cp (Chest Pain Type)**:
    - 0 = Typical angina
    - 1 = Atypical angina
    - 2 = Non-anginal pain
    - 3 = Asymptomatic
- **trestbps (Resting Blood Pressure)**: The patient’s resting blood pressure in mm Hg.
- **chol (Serum Cholesterol)**: Serum cholesterol level in mg/dl.
- **fbs (Fasting Blood Sugar)**: Whether fasting blood sugar is > 120 mg/dl (1 = true, 0 = false).
- **restecg (Resting Electrocardiographic Results)**:
    - 0 = Normal
    - 1 = ST-T wave abnormality
    - 2 = Left ventricular hypertrophy
- **thalach (Maximum Heart Rate Achieved)**: Maximum heart rate achieved during exercise.
- **exang (Exercise-Induced Angina)**: 1 = Yes, 0 = No.
- **oldpeak (ST Depression Induced by Exercise)**: ST depression relative to rest.
- **slope (Slope of the Peak Exercise ST Segment)**:
    - 0 = Upsloping
    - 1 = Flat
    - 2 = Downsloping
- **ca (Number of Major Vessels Colored by Fluoroscopy)**: Values range from 0 to 4.
- **thal (Thalassemia)**:
    - 1 = Normal
    - 2 = Fixed defect
    - 3 = Reversible defect
- **target**: The presence of heart disease (1 = Yes, 0 = No).

## Model Performance 📈
The results of the heart disease prediction models indicate that the **KNeighbors Classifier** achieved the highest accuracy score of **91.80%**, outperforming all other models. The confusion matrix for this classifier demonstrates strong predictive performance, with:
- 27 **True Negatives**
- 29 **True Positives**
- Only 2 **False Positives**
- Only 3 **False Negatives**

This suggests that the model effectively differentiates between patients with and without heart disease, with minimal misclassification.

Other models, such as **Gaussian NB** (86.89%) and **Bernoulli NB** (86.89%), also performed well, followed closely by **Logistic Regression** (85.25%) and **Random Forest Classifier** (85.25%). However, the **Gradient Boosting Classifier** (78.69%) had the lowest accuracy, indicating potential room for improvement with hyperparameter tuning or feature selection.

### Conclusion 🏁
Overall, the **KNeighbors Classifier** appears to be the most effective model for this dataset. This project demonstrates how machine learning can support early heart disease detection and improve healthcare outcomes.

## Try it yourself! 🚀
You can try the **Heart Disease Prediction Model** on Hugging Face Spaces by following this [link](https://huggingface.co/spaces/Senasu/Heart_Disease_Prediction) 🌐.
