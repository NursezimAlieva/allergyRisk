# Food Allergy Risk Predictor (AI-Powered)

An intelligent system designed to estimate the risk of 15 types of food allergies using Machine Learning and the Atopic March clinical concept.

## Project Overview
This project leverages historical medical data to predict the likelihood of food allergies in patients. The system analyzes the correlation between demographic data, birth year, and comorbid conditions (Asthma, Eczema) to provide a personalized risk assessment.

## Key Features
* 15-Allergen Prediction: Including Peanuts, Milk, Eggs, Shellfish, and a wide range of tree nuts (Cashew, Hazelnut, Pistachio, etc.).
* Explainable AI (XAI): Visualization of Feature Importance—the system explains why a specific risk level was calculated.
* Generational Analysis: A dynamic trend chart showing how birth year correlates with allergy prevalence in the dataset.
* PDF Report Generation: Users can download a formal assessment report to share with medical professionals.
* Professional ML Pipeline: Implements data scaling (StandardScaler) for high prediction stability and better convergence.

## Tech Stack
* Language: Python 3.11
* ML Libraries: Scikit-learn (Logistic Regression), Pandas, Numpy, Joblib.
* Interface: Streamlit (Web Dashboard).
* Visualization: Matplotlib.
* Reporting: FPDF.

## Data Science Implementation Details
The project incorporates several advanced data science techniques:
1. Atopic March Logic: Feature engineering based on the clinical progression from eczema and asthma to food allergies.
2. Handling Class Imbalance: Utilizes class_weight='balanced' to account for the rarity of positive allergy cases in the general population.
3. ML Pipelines: Encapsulates StandardScaler and LogisticRegression into a single object to prevent data leakage and ensure consistent preprocessing.
4. Scalability: The architecture allows for easy addition of new allergens without modifying the core UI logic.

## Getting Started

1. Clone the repository

2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app.py

## Dataset
The project is based on the open-source Zenodo (Food Allergy Analysis) dataset, containing anonymized records of thousands of patients.

---
*Disclaimer: This project is for educational purposes as part of a developer portfolio. It is not medical software and does not replace professional medical advice or diagnosis.*
