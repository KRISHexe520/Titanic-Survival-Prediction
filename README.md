![Titanic ML Banner](https://raw.githubusercontent.com/KRISHexe520/Titanic-Survival-Prediction/main/image/titanic_banner.png)
# 🚢 Titanic Survival Prediction — Machine Learning Project
![Python](https://img.shields.io/badge/Python-3.8-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Processing-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

This project predicts the survival of Titanic passengers using features such as gender, passenger class, fare, age, and family size.  
It follows a complete machine learning workflow including data cleaning, EDA, feature engineering, modeling, and evaluation.

Built using **Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn**.

---

## ⭐ Key Features
- Data preprocessing & handling missing values  
- Exploratory Data Analysis (EDA) with visual insights  
- Survival analysis by gender and passenger class  
- Correlation heatmap for feature importance  
- Logistic Regression for binary classification  
- Model evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve  
- Clean folder structure suitable for portfolio showcase  

---

## 📂 Project Structure
```
Titanic-Survival-Prediction/
├── Titanic_logistic_regression.ipynb
├── README.md
├── requirements.txt
├── data/
└── images/
```

---

# 🔍 Data Exploration & Pre-Processing

###  Steps Performed
- Loaded the dataset  
- Inspected column names and datatypes  
- Generated descriptive statistics  
- Identified missing values  
- Filled missing values:  
  - Age → median  
  - Fare → median  
  - Embarked → mode  
- Encoded categorical variables:  
  - Sex → {male:0, female:1}  
  - One-hot encoding for Embarked  
- Dropped irrelevant columns: Name, Ticket, Cabin  
- Created feature matrix (**X**) and target (**y**)  

---

# 📊 Exploratory Data Analysis (EDA)

## 1️⃣ Survival by Gender
<img width="640" height="480" src="https://github.com/user-attachments/assets/16a828d1-8a1f-4d2d-8a4c-aa34982e7df1" />

**Insights:**
- Women survived dramatically more than men  
- Gender is the strongest predictor in the dataset  

---

## 2️⃣ Survival by Passenger Class
<img width="640" height="480" src="https://github.com/user-attachments/assets/c054411c-11c8-4b6a-9515-945328737084" />

**Insights:**
- 1st class → highest survival  
- 3rd class → lowest survival  
- Wealth & cabin location influenced survival  

---

## 3️⃣ Correlation Heatmap
<img width="1000" height="600" src="https://github.com/user-attachments/assets/b8916c8d-565c-4aca-901f-060e6e78a3d0" />

**Key Findings:**
- Fare ↑ → Survival ↑  
- Pclass ↑ → Fare ↓ (strong negative correlation)  
- Pclass ↑ → Survival ↓  
- Age has weak correlation  
- SibSp & Parch correlate with each other  

---

# 🤖 Model Building — Logistic Regression

### Workflow
- Train-test split  
- Logistic Regression for binary classification  
- Predicts probability of survival  
- Evaluated using multiple metrics  

### Why Logistic Regression?
- Interpretable  
- Fast  
- Works well for binary outcomes  
- Produces probability predictions  

---

# 🧮 Model Evaluation

### Metrics
```
Accuracy  : 0.8156
Precision : 0.7015
Recall    : 0.7833
F1 Score  : 0.7401
```

### Confusion Matrix
```
[[99, 13],
 [20, 47]]
```

### ROC Curve (AUC)
<img width="640" height="480" src="https://github.com/user-attachments/assets/447994f6-d028-42f9-8035-134585fa76ca" />

---

# ⭐ Interpretation of Results

- **Gender strongly impacts survival**  
  Females had drastically higher survival rates.
- **Passenger class plays a major role**  
  1st class passengers had better access to lifeboats.
- **Fare influences survival indirectly**  
  Higher fare → higher class → higher survival.
- **Logistic Regression performed effectively**  
  Captured meaningful patterns and remained interpretable.

---

# 🏁 Conclusion

This project demonstrates a complete ML pipeline:
- Handling real-world messy data  
- Performing structured EDA  
- Extracting insights  
- Engineering useful features  
- Training & evaluating a predictive model  
- Understanding why predictions happen  

It is a strong portfolio project demonstrating foundational ML skills.

---

# 🚀 Future Improvements

- Remove outliers (Age & Fare)  
- Add advanced models: Random Forest, XGBoost, SVM  
- Hyperparameter tuning (GridSearchCV)  
- Create a `family_size` feature  
- Feature scaling  
- Deck extraction from Cabin column  

---

# 🛠 Skills Demonstrated

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis  
- Correlation & Feature Insights  
- Logistic Regression Modeling  
- Evaluation Metrics  
- ROC Analysis  
- Categorical Encoding  
- End-to-end ML pipeline development  

---

# ▶️ How to Run
```
pip install -r requirements.txt
jupyter notebook Titanic_logistic_regression.ipynb
```

---

# 👤 Author
**Krish Pandya**  
Machine Learning & AI Enthusiast  
