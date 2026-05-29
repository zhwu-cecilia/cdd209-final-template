from src.pharma_adherence.data import PharmaDataset
from src.pharma_adherence.modeling import ModelTrainer
#TODO: Import ModelTrainer class from modeling.py

"""
DAY 1: DATA PREPROCESSING & ANALYSIS
"""

dataset = PharmaDataset("data/raw/prescriptions_large_raw.csv")
print(dataset.df.head())

#TODO: Clean the dataset
dataset.clean()
#TODO: Save the dataset as a csv into "data/processed/"
dataset.save("data/processed/prescriptions_large_cleaned.csv")

#TODO: Visualize the cleaned data

"""
dataset.hist("drug_name").show()
dataset.bar("drug_name", "proportion_days_covered").show()
dataset.scatter("patient_age", "proportion_days_covered").show()
"""
#TODO: Look at the summary of a patient
patient = dataset.get_patient("P007")
print(patient.summary())
"""
DAY 2: MACHINE LEARNING
"""

#TODO: Instantiate a linear regression trainer
linear_model = ModelTrainer(dataset.df, "proportion_days_covered", ["sex", "copay_amount"])
#TODO: Train the linear model
model, metrics = linear_model.train_linear()

#TODO: Print the linear model metrics
print(metrics)
#TODO: Instantiate a logistic regression trainer

#TODO: Train the logistic model

#TODO: Print the logistic model metrics