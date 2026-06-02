from pathlib import Path
import pandas as pd
from .cleaning import clean_prescription_data
from .visualization import plot_hist, plot_bar, plot_scatter
from .patient import PatientAdherenceProfile


class PharmaDataset:
    """Wraps a prescription CSV with cleaning, visualization, and patient lookup."""

    def __init__(self, filepath):
        """Load raw prescription data from a CSV file."""
        self.df = self.load(filepath).copy()
        self.cleaned = False
    
    def load(self, filepath):
        """Read a CSV into a DataFrame, raising FileNotFoundError if the path is missing."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)
        return df

    def clean(self):
        """Run the full cleaning pipeline; must be called before visualizing or accessing patients."""
        self.df = clean_prescription_data(self.df)
        self.cleaned = True
    
    def save(self, filepath):
        """Write the cleaned DataFrame to CSV, creating parent directories as needed."""
        if self.cleaned is False:
            raise ValueError("Run clean() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(filepath)
    
    def hist(self, column_name):
        """Return a histogram of a single numeric column."""
        if self.cleaned is False:
            raise ValueError("Run clean() first.")
        
        return plot_hist(self.df, column_name, label_rotation=45)
    
    def bar(self, cat, values):
        """Return a bar chart of mean values grouped by a categorical column."""
        if self.cleaned is False:
            raise ValueError("Run clean() first.")
        
        return plot_bar(self.df, cat, values, label_rotation=45)
    
    def scatter(self, x, y):
        """Return a scatter plot of two numeric columns."""
        if self.cleaned is False:
            raise ValueError("Run clean() first.")
        
        return plot_scatter(self.df, x, y)

    def get_patient(self, patient_id):
        """Return a PatientAdherenceProfile for the given patient_id."""
        if self.cleaned is False:
            raise ValueError("Run clean() first.")
        
        patient_df = self.df[
            self.df["patient_id"] == patient_id
        ]

        return PatientAdherenceProfile(patient_id, patient_df)
    
    def get_df(self):
        """Return the underlying DataFrame."""
        return self.df
    
    def is_clean(self):
        """Return True if clean() has been called."""
        return self.cleaned