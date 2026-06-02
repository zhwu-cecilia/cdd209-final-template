import pandas as pd

class PatientAdherenceProfile:
    """Summarizes medication adherence statistics for a single patient."""

    def __init__(self, patient_id, df: pd.DataFrame):
        """Store patient fills sorted chronologically by fill date."""
        self.patient_id = patient_id
        self.df = df.sort_values("fill_date")

    def total_fills(self):
        """Return the total number of prescription fills on record."""
        #TODO: Calculate and return the total number of fills?
        return len(self.df)

    def average_copay(self):
        """Return the mean copay amount across all fills, ignoring missing values."""
        #TODO: Calculate and return the average copay_amount
        return self.df["copay_amount"].mean()

    def average_days_supply(self):
        """Return the mean days supply across all fills, ignoring missing values."""
        #TODO: Calculate and return the average days_supply
        return self.df["days_supply"].mean()

    def calculate_pdc(self):
        """Return the mean proportion of days covered (PDC) across all fills."""
        #TODO: Calculate and return the average proportion_days_covered (pdc)
        return self.df["proportion_days_covered"].mean()

    def is_adherent(self, threshold=0.75):
        """Return True if mean PDC meets or exceeds the adherence threshold (default 0.75)."""
        #TODO: Return True if average pdc meets the adherence threshold provided
        return self.calculate_pdc()>= threshold

    def summary(self):
        """Return a dict of key adherence statistics for this patient."""
        return {
            "patient_id": self.patient_id,
            "total_fills": self.total_fills(),
            "avg_copay": self.average_copay(),
            "avg_days_supply": self.average_days_supply(),
            "pdc": self.calculate_pdc(),
            "adherent": self.is_adherent(),
        }
        