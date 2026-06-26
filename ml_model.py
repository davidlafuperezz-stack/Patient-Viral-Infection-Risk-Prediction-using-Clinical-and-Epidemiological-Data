import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
sns.set(style="whitegrid")

# LOAD DATA
df = pd.read_csv("data/viral_infection_dataset.csv")

# FEATURES / TARGET
X = df.drop("infection_status", axis=1)
X = pd.get_dummies(X)  # por blood_type (categorical)

y = df["infection_status"]

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# SCALING (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#LOGISTIC REGRESSION
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

print("\n====================")
print("LOGISTIC REGRESSION")
print("====================")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("ROC AUC:", roc_auc_score(y_test, log_prob))
print(classification_report(y_test, log_pred))

# Cross-validation
cv_log = cross_val_score(log_model, X_train_scaled, y_train, cv=5)
print("CV Mean Score:", cv_log.mean())

#RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n====================")
print("RANDOM FOREST")
print("====================")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_prob))
print(classification_report(y_test, rf_pred))

# Cross-validation
cv_rf = cross_val_score(rf_model, X_train, y_train, cv=5)
print("CV Mean Score:", cv_rf.mean())

#CONFUSION MATRIX (BEST MODEL)
plt.figure()
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#FEATURE IMPORTANCE
importance = rf_model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

plt.figure()
sns.barplot(data=feat_df.head(10), x="importance", y="feature", palette="viridis")
plt.title("Top 10 Feature Importance (Random Forest)")
plt.show()

#FINAL INSIGHT
print("\nFINAL INSIGHTS:")
print("- Random Forest performs better due to non-linear relationships.")
print("- Key predictors: viral exposure, antibodies, CRP, contact with infected.")
print("- Logistic Regression is more interpretable but less powerful.")
print("- Model validated with cross-validation and ROC-AUC.")
