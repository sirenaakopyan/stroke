"""
This module provides a utilities function to assit testing.
"""
import pandas as pd


def create_sample_testing_df():
    """
    Creates a sample dataframe for testing with stroke data.

    Returns:
    A pandas DataFrame object representing the sample data.

    Example Usage:
    df = create_sample_testing_df()
    """

    return pd.DataFrame(
        {
            "id": range(5),
            "age": [30, 40, 50, 60, 70],
            "gender": ["male", "male", "female", "female", "male"],
            "hypertension": [0, 0, 1, 1, 1],
            "heart_disease": [1, 0, 0, 0, 1],
            "ever_married": ["yes", "yes", "yes", "no", "yes"],
            "work_type": [
                "private",
                "self-employed",
                "private",
                "never_worked",
                "private",
            ],
            "Residence_type": ["urban", "urban", "rural", "urban", "rural"],
            "avg_glucose_level": [120, 130, 140, 150, 160],
            "bmi": [25, 26, 27, 28, 29],
            "smoking_status": [
                "never_smoked",
                "formerly_smoked",
                "never_smoked",
                "smokes",
                "smokes",
            ],
            "stroke": [0, 1, 1, 0, 1],
        }
    )
