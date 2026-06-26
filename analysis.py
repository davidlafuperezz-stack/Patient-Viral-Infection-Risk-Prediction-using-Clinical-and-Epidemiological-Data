import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CONFIG
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# LOAD DATA
df = pd.read_csv("data/viral_infection_dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())

# 1. TARGET DISTRIBUTION
plt.figure()
sns.countplot(x="infection_status", data=df)
plt.title("Infection Distribution (0 = No, 1 = Yes)")
plt.show()

# 2. AGE vs INFECTION
plt.figure()
sns.boxplot(x="infection_status", y="age", data=df)
plt.title("Age vs Infection Status")
plt.show()

# 3. BMI vs INFECTION
plt.figure()
sns.boxplot(x="infection_status", y="bmi", data=df)
plt.title("BMI vs Infection Status")
plt.show()

# 4. VIRAL EXPOSURE vs INFECTION
plt.figure()
sns.boxplot(x="infection_status", y="viral_exposure", data=df)
plt.title("Viral Exposure vs Infection")
plt.show()

# 5. BIOMARKERS
plt.figure()
sns.boxplot(x="infection_status", y="crp", data=df)
plt.title("CRP Levels vs Infection Status")
plt.show()

plt.figure()
sns.boxplot(x="infection_status", y="antibodies", data=df)
plt.title("Antibody Levels vs Infection Status")
plt.show()

plt.figure()
sns.boxplot(x="infection_status", y="oxygen_sat", data=df)
plt.title("Oxygen Saturation vs Infection Status")
plt.show()

# 6. CORRELATION HEATMAP
plt.figure()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# 7. FEATURE IMPACT (IMPORTANT GRAPH)
corr_target = df.corr(numeric_only=True)["infection_status"].sort_values(ascending=False)
corr_target = corr_target.drop("infection_status")

plt.figure()
sns.barplot(x=corr_target.values, y=corr_target.index, palette="viridis")
plt.title("Feature Impact on Infection Risk")
plt.xlabel("Correlation with Infection")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# INSIGHTS
print("\nINSIGHTS:")
print("- Higher viral exposure increases infection risk.")
print("- Lower antibody levels correlate with infection.")
print("- CRP (inflammation marker) is higher in infected patients.")
print("- Age and comorbidities contribute moderately to risk.")
print("- Feature impact plot summarizes key drivers of infection.")
