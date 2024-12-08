# Loan Approval Prediction Using Machine Learning

This project demonstrates a machine learning pipeline for predicting loan approval using a real-world dataset. The focus is on preprocessing, feature transformation, handling imbalanced data, training, and hyperparameter tuning of models like `LGBMClassifier`.

---

## **Project Overview**
The project implements the following key steps:
1. **Data Preprocessing**:
   - Handling missing values.
   - Transforming skewed features using log, square root, and Yeo-Johnson transformations.
   - Encoding categorical features using appropriate techniques (e.g., one-hot encoding, ordinal mapping).

2. **Exploratory Data Analysis (EDA)**:
   - Distribution plots for features.
   - Target variable analysis (handling class imbalance).
   - Correlation matrix to identify and drop highly correlated features.

3. **Model Training**:
   - Implements multiple models using `LazyPredict` for initial benchmarking.
   - Focus on `LGBMClassifier` with hyperparameter tuning using `RandomizedSearchCV` and 5-fold cross-validation.

4. **Performance Evaluation**:
   - Metrics include confusion matrix, classification report, precision, recall, F1-score, and AUC-ROC curve.

---

## **Dataset**
The dataset includes various numerical and categorical features related to loan applications. Key columns include:
- **ApplicationDate**: Date of loan application.
- **EmploymentStatus**: Categorical values like `Employed`, `Self-Employed`.
- **EducationLevel**: Ordinal values like `Master`, `Bachelor`, etc.
- **LoanAmount**, **AnnualIncome**, and more.

The target variable is **LoanApproved**:
- `0`: Loan not approved.
- `1`: Loan approved.

---

## **How to Run the Project**

1. Clone this repository:
   ```bash
   git clone https://github.com/<username>/<repository>.git
   cd <repository>
