import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

'''
-------------------------------------------------------------------------------
 1. Generate dataset (toy version of hospital readmissions)
-------------------------------------------------------------------------------
'''
np.random.seed(42)
n = 5000

data = pd.DataFrame({
    "age": np.random.randint(20, 90, n),
    "num_prev_admissions": np.random.poisson(1.5, n),
    "length_of_stay": np.random.randint(1, 15, n),
    "diabetes": np.random.binomial(1, 0.2, n),
    "chf": np.random.binomial(1, 0.1, n),   # congestive heart failure
    "creatinine": np.random.normal(1.0, 0.3, n),  # kidney function
    "insurance_type": np.random.choice(["Medicare", "Medicaid", "Private"], n)
})

# Target: readmitted within 30 days (skewed outcome)
data["readmitted_30d"] = (
    0.1*data["num_prev_admissions"] +
    0.2*data["diabetes"] +
    0.3*data["chf"] +
    0.1*(data["creatinine"] > 1.2) +
    np.random.binomial(1, 0.05, n)   # noise
) > 0.5
data["readmitted_30d"] = data["readmitted_30d"].astype(int)

'''
-------------------------------------------------------------------------------
 2. Preprocessing
-------------------------------------------------------------------------------
'''
# One-hot encode categorical variable (insurance type)
data = pd.get_dummies(data, columns =["insurance_type"],drop_first=True)

X = data.drop("readmitted_30d",axis=1)
y = data["readmitted_30d"]
X_train, X_test, y_train, y_test =train_test_split(
    X,y, test_size=0.2, stratify=y,random_state=42)

'''
-------------------------------------------------------------------------------
 3. Train XGBoost
-------------------------------------------------------------------------------
'''
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="logloss")

model.fit(X_train,y_train)

'''
-------------------------------------------------------------------------------
 4. Evaluate
-------------------------------------------------------------------------------
'''
y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba > 0.5).astype(int)

print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred))

'''
-------------------------------------------------------------------------------
 5. Interpret feature importance
-------------------------------------------------------------------------------
'''

# Set default text color to white
plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 
                     'xtick.color': 'white', 'ytick.color': 'white'})

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_facecolor('#2e2e2e')  # figure background
ax.set_facecolor('#2e2e2e')         # axes background

# Plot feature importance; bar color
xgb.plot_importance(model, ax=ax, importance_type='gain', color='#ffa500')

ax.set_title("Feature Importance (Gain)")
plt.show()

# Get importance as a DataFrame
importance_df = pd.DataFrame({
    'feature': list(model.get_booster().get_score(importance_type='gain').keys()),
    'importance': list(model.get_booster().get_score(importance_type='gain').values())
})

# Sort descending
importance_df = importance_df.sort_values(by='importance', ascending=False)


'''
-------------------------------------------------------------------------------
 6. Convert to risk groups
-------------------------------------------------------------------------------
'''
# Create DataFrame to store patient IDs and preidcted probability
risk_df =pd.DataFrame({
    'patient_id': X_test.index,
    'readmit_proba' : y_pred_proba,
    })

# Quantile-based split (tertiles)
risk_df['risk_group'] = pd.qcut(risk_df['readmit_proba'], q=3, labels =['Low','Medium', 'High'])

# Preview
print(risk_df.head())
print(risk_df['risk_group'].value_counts())


'''
-------------------------------------------------------------------------------
 7. Export data to Excel
-------------------------------------------------------------------------------
'''
# Feature importance
importance_df.to_excel('feature_importance.xlsx', index=False)

# Risk groups
risk_df.to_excel('patient_risk.xlsx', index=False)

print("Export complete: feature_importance.xlsx and patient_risk.xlsx")



















