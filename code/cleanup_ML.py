import pandas as pd


def risk_factor_df_ML(dataframe: str) -> pd.DataFrame:

    over_65 = (dataframe.over_65 == 0) | (dataframe.over_65 == 1)
    gender = (dataframe.gender == 0) | (dataframe.gender == 1)
    smoking_status = (dataframe.smoking_status == 0) |\
                     (dataframe.smoking_status == 1)
    low_bmi = (dataframe.low_BMI == 0) | (dataframe.low_BMI == 1)
    high_bmi = (dataframe.high_BMI == 0) | (dataframe.high_BMI == 1)
    high_glucose = (dataframe.high_glucose == 0) |\
                   (dataframe.high_glucose == 1)
    residence = (dataframe.residence == 0) | (dataframe.residence == 1)
    hypertension = (dataframe.hypertension == 0) |\
                   (dataframe.hypertension == 1)
    heart_disease = (dataframe.heart_disease == 0) |\
                    (dataframe.heart_disease == 1)
    married = (dataframe.married == 0) | (dataframe.married == 1)
    stroke = (dataframe.stroke == 0) | (dataframe.stroke == 1)
    ml_df = dataframe[over_65 & gender & smoking_status & low_bmi &
                      high_bmi & high_glucose & residence &
                      hypertension & heart_disease & married & stroke]
    return ml_df
