import dataanalysis
import pandas as pd
import numpy as np

def risk_factor_df_ML(dataframe: str) -> pd.DataFrame:
    correlations = dataanalysis.find_correlations(dataframe)    
    over_65 = correlations['over_65']
    heart_disease = correlations['heart_disease']
    hypertension = correlations['hypertension']
    married = correlations['married']
    low_bmi = correlations['low_BMI']
    residence = correlations['residence']
    high_BMI = correlations['high_BMI']
    high_glucose = correlations['high_glucose']
    gender = correlations['gender']
    residence = correlations['residence']
    smoking_status = correlations['smoking_status']
    dataframe2 = dataframe.copy()
    dataframe2 = dataframe2[['hypertension', 'heart_disease',
                             'high_glucose','low_BMI','high_BMI', 
                             'over_65', 'married', 'residence', 'gender', 'smoking_status']]
    dataframe3 = dataframe2.apply(pd.Series.value_counts, axis=1)
    dataframe2["hypertension"] = np.where(dataframe2["hypertension"] == 1,
                                          hypertension, 0)
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
    dataframe2['gender'] = np.where(dataframe2['gender'] == 1, gender, 0)
    dataframe2['smoking_status'] = np.where(dataframe2['smoking_status'] == 1, smoking_status, 0)
    dataframe4 = dataframe2.copy()
    dataframe4.loc[:, 'row_corr'] = dataframe2.sum(axis=1)
    dataframe4 = dataframe4[["row_corr"]]
    row_corr_and_one_count = pd.merge(dataframe3, dataframe4,left_index=True, right_index=True)
    row_corr_and_one_count = row_corr_and_one_count[[1.0, "row_corr"]]
    row_corr_and_one_count['risk_corr'] = (row_corr_and_one_count['row_corr'] / row_corr_and_one_count[1.0])
    dataframe5 = row_corr_and_one_count[['risk_corr']]
    mL_df = pd.merge(dataframe, dataframe5, left_index=True, right_index=True)
    mL_df = mL_df[["stroke", "risk_corr"]]
    mL_df = mL_df.fillna(0)
    return mL_df
    




