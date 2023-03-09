import datacleanup
import dataanalysis
import pandas as pd
import numpy as np

def risk_factor_df_ML(dataframe: str) -> pd.DataFrame:
    correlations = dataanalysis.find_risk_factor_correlation(dataframe)
    over_65 = correlations[1]
    heart_disease = correlations[2]
    hypertension = correlations[3]
    married = correlations[4]
    low_bmi = correlations[5]
    residence = correlations[6]
    high_BMI = correlations[7]
    high_glucose = correlations[8]
    dataframe2 = dataframe.copy()
    dataframe2 = dataframe2[['hypertension', 'heart_disease', 'high_glucose','low_BMI',
                             'high_BMI', 'over_65', 'married', 'residence']]
    dataframe2["hypertension"] = np.where(dataframe2["hypertension"] == 1, hypertension, 0)
    dataframe2['heart_disease'] = np.where(dataframe2['heart_disease'] == 1,
                                           heart_disease, 0)
    dataframe2['high_glucose'] = np.where(dataframe2['high_glucose'] == 1,
                                          high_glucose, 0)
    dataframe2['low_BMI'] = np.where(dataframe2['low_BMI'] == 1, low_bmi, 0)
    dataframe2['high_BMI'] = np.where(dataframe2['high_BMI'] == 1, high_BMI,
                                       0)
    dataframe2['over_65'] = np.where(dataframe2['over_65'] == 1, over_65,
                                     0)
    dataframe2['married'] = np.where(dataframe2['married'] == 1, married,
                                     0)
    dataframe2['residence'] = np.where(dataframe2['residence'] == 1, residence,
                                       0)
    dataframe = dataframe[['stroke']]
    mL_df = pd.merge(dataframe, dataframe2, left_index=True, right_index=True)
    # mL_df = mL_df[['stroke', 'row_corr']]
    return mL_df




