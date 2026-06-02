"""End-to-end demo: load and clean prescription data, visualize, profile a patient, and train ML models."""
from src.pharma_adherence.data import PharmaDataset
from src.pharma_adherence.modeling import ModelTrainer
#TODO: Import ModelTrainer class from modeling.py

"""
DAY 1: DATA PREPROCESSING & ANALYSIS
"""

# Load raw data and preview the first few rows
dataset = PharmaDataset("data/raw/prescriptions_large_raw.csv")
print(dataset.df.head())

#TODO: Clean the dataset
dataset.clean()
#TODO: Save the dataset as a csv into "data/processed/"
dataset.save("data/processed/prescriptions_large_cleaned.csv")

#TODO: Visualize the cleaned data


dataset.hist("drug_name").show()
dataset.bar("drug_name", "proportion_days_covered").show()
dataset.scatter("patient_age", "proportion_days_covered").show()

#TODO: Look at the summary of a patient
patient = dataset.get_patient("P007")
print(patient.summary())
"""
DAY 2: MACHINE LEARNING
"""

#TODO: Instantiate a linear regression trainer
# Predict the continuous PDC score from sex and copay
linear_model = ModelTrainer(dataset.df, "proportion_days_covered", ["sex", "copay_amount"])

#TODO: Train the linear model
model, metrics = linear_model.train_linear()

#TODO: Print the linear model metrics
print(metrics)

#TODO: Instantiate a logistic regression trainer
# Predict the binary adherence flag (0/1) from the same features
logistic_model = ModelTrainer(dataset.df, "adherence_flag", ["sex", "copay_amount"])

#TODO: Train the logistic model
model, metrics = logistic_model.train_logistic()

#TODO: Print the logistic model metrics
print(metrics)