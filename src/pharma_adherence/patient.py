import pandas as pd

class PatientAdherenceProfile:
    def __init__(self, patient_id, df: pd.DataFrame):
        self.patient_id = patient_id
        self.df = df.sort_values("fill_date")

    def total_fills(self):
        #TODO: Calculate and return the total number of fills?
        return len(self.df)

    def average_copay(self):
        #TODO: Calculate and return the average copay_amount
        return self.df["copay_amount"].mean()

    def average_days_supply(self):
        #TODO: Calculate and return the average days_supply
        return self.df["days_supply"].mean()

    def calculate_pdc(self):
        #TODO: Calculate and return the average proportion_days_covered (pdc)
        return self.df["proportion_days_covered"].mean()

    def is_adherent(self, threshold=0.75):
        #TODO: Return True if average pdc meets the adherence threshold provided
        return self.calculate_pdc()>= threshold

    def summary(self):
        return {
            "patient_id": self.patient_id,
            "total_fills": self.total_fills(),
            "avg_copay": self.average_copay(),
            "avg_days_supply": self.average_days_supply(),
            "pdc": self.calculate_pdc(),
            "adherent": self.is_adherent(),
        }
        