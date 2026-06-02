import numpy as np
import pandas as pd
import pandas.testing as pdt

from pharma_adherence.cleaning import clean_prescription_data

"""
RUN THIS SCRIPT USING `PYTHONPATH=src pytest -v`
"""

def test_clean_prescription_data_basic_cleaning():
    df = pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P001"],
            "fill_date": ["2024-12-30", "Jan 26 2025", "Mar 02 2025"],
            "drug_name": ["amoxicillin", "PANTOPRAZOLE", "AMLODIPINE 5MG"],
            "days_supply": ["60", "60", "30days"],
            "quantity_dispensed": ["60", "120", "30"],
            "refill_number": ["0", "", ""],
            "patient_age": ["34", "34", "34"],
            "sex": ["f", "f", "f"],
            "zip_code": ["94720", "94720", "94720"],
            "prescriber_id": ["D123", "D208", "D217"],
            "pharmacy_name": ["walgreens pharmacy", "cvs pharmacy", "Walgreens"],
            "copay_amount": ["19.73", "9", "12.6"],
            "adherence_flag": ["0", "1", "0"],
            "proportion_days_covered": ["0.57", "0.78", "0.65"],
        }
    )

    result = clean_prescription_data(df)

    expected = pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P001"],
            "fill_date": pd.to_datetime(
                ["2024-12-30", "2025-01-26", "2025-03-02"]
            ),
            "drug_name": ["amoxicillin", "pantoprazole", "amlodipine"],
            "days_supply": [60.0, 60.0, 30.0],
            "quantity_dispensed": [60.0, 120.0, 30.0],
            "refill_number": [0.0, np.nan, np.nan],
            "patient_age": [34.0, 34.0, 34.0],
            "sex": ["female", "female", "female"],
            "zip_code": ["94720", "94720", "94720"],
            "prescriber_id": ["D123", "D208", "D217"],
            "pharmacy_name": ["walgreens", "cvs", "walgreens"],
            "copay_amount": [19.73, 9.0, 12.6],
            "adherence_flag": [0.0, 1.0, 0.0],
            "proportion_days_covered": [0.57, 0.78, 0.65],
        }
    )

    pdt.assert_frame_equal(result, expected, check_dtype=False)


def test_clean_prescription_data_drops_bad_rows():
    df = pd.DataFrame(
        {
            "patient_id": ["P001", "", "P002"],
            "fill_date": ["2024-12-30", "2025-01-01", "bad date"],
            "drug_name": ["amoxicillin", "pantoprazole", "amlodipine"],
            "days_supply": ["60", "60", "60"],
            "quantity_dispensed": ["60", "120", "30"],
            "refill_number": ["0", "0", "0"],
            "patient_age": ["34", "34", "34"],
            "sex": ["f", "f", "f"],
            "zip_code": ["94720", "94720", "94720"],
            "prescriber_id": ["D123", "D208", "D217"],
            "pharmacy_name": ["walgreens pharmacy", "cvs pharmacy", "Walgreens"],
            "copay_amount": ["19.73", "9", "12.6"],
            "adherence_flag": ["0", "1", "0"],
            "proportion_days_covered": ["0.57", "0.78", "0.65"],
        }
    )

    result = clean_prescription_data(df)

    # Only the first row is fully valid
    assert len(result) == 1
    assert result.iloc[0]["patient_id"] == "P001"
    assert result.iloc[0]["drug_name"] == "amoxicillin"


def test_clean_prescription_data_handles_out_of_range_values():
    df = pd.DataFrame(
        {
            "patient_id": ["P001"],
            "fill_date": ["2024-12-30"],
            "drug_name": ["AMLODIPINE 5MG"],
            "days_supply": ["0"],
            "quantity_dispensed": ["-1"],
            "refill_number": ["-2"],
            "patient_age": ["150"],
            "sex": ["m"],
            "zip_code": ["94720"],
            "prescriber_id": ["D123"],
            "pharmacy_name": ["walgreens pharmacy"],
            "copay_amount": ["-5"],
            "adherence_flag": ["3"],
            "proportion_days_covered": ["1.2"],
        }
    )

    result = clean_prescription_data(df)

    assert pd.isna(result.loc[0, "days_supply"])
    assert pd.isna(result.loc[0, "quantity_dispensed"])
    assert pd.isna(result.loc[0, "refill_number"])
    assert pd.isna(result.loc[0, "patient_age"])
    assert pd.isna(result.loc[0, "copay_amount"])
    assert pd.isna(result.loc[0, "adherence_flag"])
    assert pd.isna(result.loc[0, "proportion_days_covered"])
    assert result.loc[0, "sex"] == "male"
    assert result.loc[0, "drug_name"] == "amlodipine"


def test_clean_prescription_data_drops_rows_with_missing_critical_fields():
    df = pd.DataFrame(
        {
            "patient_id": ["P001", None, "P002"],
            "fill_date": ["2024-12-30", "2025-01-01", "bad date"],
            "drug_name": ["amoxicillin", "pantoprazole", "amlodipine"],
            "days_supply": ["60", "60", "60"],
            "quantity_dispensed": ["60", "120", "30"],
            "refill_number": ["0", "0", "0"],
            "patient_age": ["34", "34", "34"],
            "sex": ["f", "f", "f"],
            "zip_code": ["94720", "94720", "94720"],
            "prescriber_id": ["D123", "D208", "D217"],
            "pharmacy_name": ["walgreens pharmacy", "cvs pharmacy", "Walgreens"],
            "copay_amount": ["19.73", "9", "12.6"],
            "adherence_flag": ["0", "1", "0"],
            "proportion_days_covered": ["0.57", "0.78", "0.65"],
        }
    )

    result = clean_prescription_data(df)

    assert len(result) == 1
    assert result.iloc[0]["patient_id"] == "P001"