# Predicting 30-Day Hospital Readmissions with XGBoost

## Project Overview
This project develops a predictive model to estimate the risk of 30-day hospital readmission for patients. Using clinical, demographic, and hospital data, the model stratifies patients into risk groups to guide targeted interventions and optimize resource allocation.

## Data
- **Patient demographics:** age, sex, ZIP code, insurance type  
- **Clinical features:** diagnoses, lab results, vital signs, comorbidities  
- **Hospitalization history:** prior admissions, length of stay  
- **Target:** readmitted within 30 days (binary outcome)  

*Note: Synthetic or anonymized patient data used for demonstration.*

## Methodology
1. **Data preprocessing**  
   - One-hot encoding of categorical variables  
   - Train-test split with stratification to preserve class balance  

2. **Modeling**  
   - XGBoost (`XGBClassifier`) trained to predict readmission probability  
   - Hyperparameters tuned for regularization, tree depth, and subsampling  

3. **Risk stratification**  
   - Patients divided into **Low, Medium, High** risk groups using probability tertiles  

4. **Explainability**  
   - Feature importance extracted from the trained model  
   - Optional: SHAP values for patient-level interpretability  

## Outputs
- **Excel files**:  
  - `feature_importance.xlsx` → importance of each feature by gain  
  - `patient_risk.xlsx` → patient ID, predicted probability, and risk group  
- **Visualizations:** feature importance plots (customizable dark background)  

## Applications in Healthcare Consulting
- Targeted care management for high-risk patients  
- Data-driven resource allocation  
- Identification of key risk factors for policy and intervention recommendations  
- Quantification of potential cost savings and operational impact  

## Requirements
- Python 3.7+  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`

## Usage
1. Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
