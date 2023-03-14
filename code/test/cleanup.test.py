"""
This module contain testing function use for datacleanup module
"""
from io import StringIO
import os
import sys
import pandas as pd
import numpy as np
import datacleanup
from scipy.stats import pearsonr

# getting current directory
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
# import the module with relative path


def test_create_risk_factor_df():
    """
    Testing function for  create_risk_factor_df
    """
    # create sample data
    sample_data = StringIO(
        """id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
        1,Male,67,0,1,Yes,Private,Urban,228.69,36.6,formerly smoked,1
        2,Female,61,0,0,Yes,Self-employed,Rural,202.21,28.1,never smoked,1
        3,Male,80,0,1,Yes,Private,Rural,105.92,32.5,never smoked,1
        4,Female,49,0,0,Yes,Private,Urban,171.23,34.4,smokes,1
        5,Female,79,1,0,Yes,Self-employed,Rural,174.12,24,never smoked,1
        """
    )

    # convert sample data to binary
    df = datacleanup.create_risk_factor_df(sample_data)

    # check if hypertension is converted correctly
    assert df.loc[0, "hypertension"] == 0
    assert df.loc[1, "hypertension"] == 0
    assert df.loc[2, "hypertension"] == 0
    assert df.loc[3, "hypertension"] == 0
    assert df.loc[4, "hypertension"] == 1

    # check if heart disease is converted correctly
    assert df.loc[0, "heart_disease"] == 1
    assert df.loc[1, "heart_disease"] == 0
    assert df.loc[2, "heart_disease"] == 1
    assert df.loc[3, "heart_disease"] == 0
    assert df.loc[4, "heart_disease"] == 0

    # check if high glucose is converted correctly
    assert df.loc[0, "high_glucose"] == 1
    assert df.loc[1, "high_glucose"] == 1
    assert df.loc[2, "high_glucose"] == 1
    assert df.loc[3, "high_glucose"] == 1
    assert df.loc[4, "high_glucose"] == 1

    # check if low BMI is converted correctly
    assert df.loc[0, "low_BMI"] == 0
    assert df.loc[1, "low_BMI"] == 0
    assert df.loc[2, "low_BMI"] == 0
    assert df.loc[3, "low_BMI"] == 0
    assert df.loc[4, "low_BMI"] == 0

    # check if high BMI is converted correctly
    assert df.loc[0, "high_BMI"] == 0
    assert df.loc[1, "high_BMI"] == 1
    assert df.loc[2, "high_BMI"] == 0
    assert df.loc[3, "high_BMI"] == 0
    assert df.loc[4, "high_BMI"] == 0

    # check if gender is converted correctly
    assert df.loc[0, "gender"] == 1
    assert df.loc[1, "gender"] == 0
    assert df.loc[2, "gender"] == 1


def test_data_transformation_correlation():
    """
    Function to test the correlation transformation of a df
    """
    # Generate sample data with correlations between age, glucose level, bmi
    np.random.seed(42)
    age = np.random.normal(40, 10, 1000)
    glucose = age * 0.2 + np.random.normal(100, 20, 1000)
    bmi = age * 0.1 + glucose * 0.3 + np.random.normal(25, 5, 1000)
    data = pd.DataFrame({"age": age, "avg_glucose_level": glucose, "bmi": bmi})

    # Call the function to transform the data
    transformed_data = datacleanup.data_transformation_correlation(data)

    # Check that the transformed data has the same shape as the original data
    assert transformed_data.shape == data.shape

    # Check that the cor between age, glucose level, and bmi are preserved
    transformed_age = transformed_data["age"]
    transformed_glucose = transformed_data["avg_glucose_level"]
    transformed_bmi = transformed_data["bmi"]
    age_glucose_corr, _ = pearsonr(transformed_age, transformed_glucose)
    glucose_bmi_corr, _ = pearsonr(transformed_glucose, transformed_bmi)
    age_bmi_corr, _ = pearsonr(transformed_age, transformed_bmi)
    assert abs(age_glucose_corr - 0.2) > 0.1
    assert abs(glucose_bmi_corr - 0.3) > 0.1
    assert abs(age_bmi_corr - 0.1) < 0.1

    # Check that all columns have been label encoded
    for col in transformed_data.columns:
        assert (
            transformed_data[col].dtype == "int64"
            or transformed_data[col].dtype == "float64"
        )

    # Check that numerical features have been standardized
    assert abs(transformed_age.mean()) < 0.1
    assert abs(transformed_age.std() - 1) < 0.1
    assert abs(transformed_glucose.mean()) < 0.1
    assert abs(transformed_glucose.std() - 1) < 0.1
    assert abs(transformed_bmi.mean()) < 0.1
    assert abs(transformed_bmi.std() - 1) < 0.1


def main():
    """
    Main function to call all test
    """
    test_create_risk_factor_df()
    test_data_transformation_correlation()


if __name__ == "__main__":
    main()
