import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

# DEMOGRAPHICS
age = np.random.randint(0, 90, n)
sex = np.random.binomial(1, 0.5, n)
bmi = np.clip(np.random.normal(25, 5, n), 15, 45)

smoking = np.random.binomial(1, 0.25, n)
alcohol = np.random.binomial(1, 0.3, n)

blood_type = np.random.choice(["A", "B", "AB", "O"], n)

# EXPOSURE
viral_exposure = np.random.randint(0, 11, n)
contact_infected = np.random.binomial(1, 0.3, n)
days_exposed = np.random.randint(0, 31, n)
travel_history = np.random.binomial(1, 0.2, n)
crowded_env = np.random.binomial(1, 0.4, n)
healthcare_worker = np.random.binomial(1, 0.1, n)

# CLINICAL
chronic_diseases = np.random.randint(0, 4, n)
diabetes = np.random.binomial(1, 0.15, n)
hypertension = np.random.binomial(1, 0.2, n)
immunosuppressed = np.random.binomial(1, 0.08, n)
vaccinated = np.random.binomial(1, 0.6, n)

# BIOMARKERS
crp = np.abs(np.random.normal(5, 10, n))
antibodies = np.random.normal(50, 20, n)
temperature = np.random.normal(36.8, 0.7, n)
oxygen_sat = np.clip(np.random.normal(96, 3, n), 80, 100)

# EPIDEMIOLOGY
population_density = np.random.choice([0, 1, 2], n)
region_risk = np.random.randint(0, 11, n)
hospital_exposure = np.random.binomial(1, 0.15, n)
outbreak_region = np.random.binomial(1, 0.25, n)
quarantine_compliance = np.random.randint(0, 11, n)


# RISK MODEL
risk = (
    0.03 * age +
    2 * contact_infected +
    0.5 * viral_exposure +
    1.5 * crowded_env +
    1.2 * immunosuppressed +
    0.8 * diabetes +
    0.6 * hypertension +
    0.5 * smoking +
    0.02 * crp -
    0.02 * antibodies -
    1.5 * vaccinated +
    2 * outbreak_region
)

risk += np.random.normal(0, 2, n)

prob = 1 / (1 + np.exp(-0.3 * (risk - 5)))

infection_status = np.random.binomial(1, prob)

# DATAFRAME
df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "smoking": smoking,
    "alcohol": alcohol,
    "blood_type": blood_type,

    "viral_exposure": viral_exposure,
    "contact_infected": contact_infected,
    "days_exposed": days_exposed,
    "travel_history": travel_history,
    "crowded_env": crowded_env,
    "healthcare_worker": healthcare_worker,

    "chronic_diseases": chronic_diseases,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "immunosuppressed": immunosuppressed,
    "vaccinated": vaccinated,

    "crp": crp,
    "antibodies": antibodies,
    "temperature": temperature,
    "oxygen_sat": oxygen_sat,

    "population_density": population_density,
    "region_risk": region_risk,
    "hospital_exposure": hospital_exposure,
    "outbreak_region": outbreak_region,
    "quarantine_compliance": quarantine_compliance,

    "infection_status": infection_status
})

# SAVE CSV
df.to_csv("viral_infection_dataset.csv", index=False)

print("Dataset created:", df.shape)
