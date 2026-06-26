# Patient-Viral-Infection-Risk-Prediction-using-Clinical-and-Epidemiological-Data
Machine learning project for analyzing viral infection risk using synthetic clinical, immunological, and epidemiological data. Includes exploratory data analysis, visualization, and predictive modeling to identify key factors associated with infection status.
# 🧬 Patient Viral Infection Risk Prediction using Clinical and Epidemiological Data

## 📌 Overview
Machine learning project for analyzing viral infection risk using synthetic clinical, immunological, and epidemiological data.

The goal is to identify key factors associated with infection status and build predictive models capable of estimating infection risk.

This project includes exploratory data analysis (EDA), data visualization, and machine learning models for classification.

##Objective
- Predict whether a patient is infected (0/1)
- Analyze clinical and epidemiological risk factors
- Compare different machine learning models
- Evaluate performance using standard classification metrics

##Dataset Description
The dataset is synthetically generated to simulate realistic biomedical data and includes:

###Demographics
- Age, sex, BMI
- Smoking and alcohol use
- Blood type

###Exposure factors
- Viral exposure level
- Contact with infected individuals
- Crowded environments
- Travel history

###Clinical biomarkers
- CRP (inflammation marker)
- Antibody levels
- Body temperature
- Oxygen saturation

###Epidemiological factors
- Population density
- Regional risk level
- Hospital exposure
- Outbreak region
- Quarantine compliance

## Machine Learning Models
- Logistic Regression (baseline, interpretable model)
- Random Forest Classifier (non-linear relationships, higher performance)


##Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Cross-validation
- Confusion matrix

##Visualizations
- Infection distribution analysis
- Feature distributions (age, BMI, biomarkers)
- Correlation heatmap
- Feature importance plot
- ROC curve

##Key Insights
- Viral exposure and contact history are strong predictors of infection
- Antibody levels are negatively correlated with infection risk
- CRP levels increase in infected patients
- Random Forest performs better than Logistic Regression

##Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

##Project Structure
data/
  viral_infection_dataset.csv
generate_viral_dataset.py
analysis.py
ml_model.py
README.md
##Notes
This project is for educational and portfolio purposes. The dataset is synthetic but designed to reflect realistic biomedical relationships.

##Author
David Lafuente Pérez. Biomedical background with ongoing training in Machine Learning and AI applications in healthcare.
