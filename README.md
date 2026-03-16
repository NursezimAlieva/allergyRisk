# Food Allergy Risk Prediction using Machine Learning

## Project Overview

This project develops a machine learning system that predicts the probability of developing common food allergies based on demographic and medical features. The model takes patient information such as age, gender, ethnicity, asthma history, and eczema history, and estimates the risk of several allergies.

The goal of this project is to demonstrate how machine learning can be used in healthcare data analysis to support early risk assessment for allergic diseases.

The model currently predicts the probability of the following allergies:

* Peanut allergy
* Milk allergy
* Egg allergy

---

## Problem Statement

Food allergies affect a significant portion of the population and can lead to serious health complications. Early detection and risk estimation are important for prevention and clinical decision making.

However, identifying allergy risk can be difficult when relying only on observable symptoms. Machine learning models can analyze patient data and identify patterns associated with increased allergy risk.

This project explores whether machine learning models can predict allergy risk using available demographic and medical features.

---

## Dataset

The project uses a publicly available food allergy dataset containing demographic and medical information about patients and the age at which allergies were diagnosed.

Key variables in the dataset include:

* **AGE_START_YEARS** – age of the patient
* **GENDER** – gender category
* **RACE** – race category
* **ETHNICITY** – ethnicity category
* **ASTHMA_START** – age when asthma began
* **ATOPIC_DERM_START** – age when eczema began
* **PEANUT_ALG_START** – age when peanut allergy began
* **MILK_ALG_START** – age when milk allergy began
* **EGG_ALG_START** – age when egg allergy began

Allergy onset columns were converted into binary variables:

* 0 = no allergy
* 1 = allergy present

---

## Data Preprocessing

Several preprocessing steps were performed before training the models:

1. Converting allergy onset age into binary classification labels.
2. Creating health indicator variables:

   * **ASTHMA_PRESENT**
   * **ECZEMA_PRESENT**
3. Handling missing values.
4. Encoding categorical variables using one-hot encoding.

These steps ensure that the dataset is suitable for machine learning algorithms.

---

## Machine Learning Models

Several classification algorithms were tested and compared:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)

The models were evaluated using **ROC-AUC score**, which measures how well the model distinguishes between allergy and non-allergy cases.

After evaluation, **Logistic Regression** performed best and was selected as the final model.

---

## Multi-Allergy Prediction

To predict multiple allergies simultaneously, the project uses a multi-output classification approach.

The model is implemented using a multi-output wrapper that trains a separate classifier for each allergy while sharing the same input features.

This allows the system to produce a list of allergy probabilities for a single patient input.

Example output:

```
{
  "peanut_allergy_risk": 0.17,
  "milk_allergy_risk": 0.10,
  "egg_allergy_risk": 0.14
}
```

---

## Project Structure

```
project/
│
├── data/
│   └── foodAllergyAnalysisZenodo.csv
│
├── notebooks/
│   └── main.ipynb
│   └── peanut.ipynb
│   └── milk.ipynb
│   └── egg.ipynb
│
└──  README.md
```

---

## Example Prediction

Example patient input:

| Feature   | Value        |
| --------- | ------------ |
| Age       |       1      |
| Gender    |     Male     |
| Race      |     White    |
| Ethnicity | Non-hispanic |
| Asthma    |      Yes     |
| Eczema    |      Yes     |

Predicted risks:

* Peanut allergy risk: 17%
* Milk allergy risk: 10%
* Egg allergy risk: 14%

---

## Limitations

This project has several limitations:

* Limited number of features available in the dataset
* Class imbalance (allergy cases are relatively rare)
* Dataset may not represent all populations

Future improvements could include additional medical history variables and larger datasets.

---

## Future Work

Possible improvements include:

* Feature importance analysis
* Advanced models such as gradient boosting or neural networks
* Hyperparameter tuning
* Larger and more diverse datasets

---

## Technologies Used

* Python
* pandas
* scikit-learn
* NumPy
* Jupyter Notebook
* matplotlip pyplot

---

## Author

Nursezim Alieva
