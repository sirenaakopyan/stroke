'''
this file contains functions that clean up the csv dataset
and attempt to find the top 3 demographic and health risk
factors for stroke.
'''
import pandas as pd
import numpy as np
import geopandas as gpd
from functools import reduce
import dataanalysis

unclean_data = pd.read_csv('datasets/stroke_data_1.csv')

'''
cleaning data in gender column. Male = 1 and Female = 0
we do not worry about other genders because they are not
included in the dataset
'''
cleaned_gender = unclean_data.copy()
cleaned_gender.loc[cleaned_gender["gender"] == "Male", "gender"] = 1
cleaned_gender.loc[cleaned_gender["gender"] == "Female", "gender"] = 0
'''cleaning data in smoking_status column. has smoked = 1.
Never smoked = 0. UNSURE WHAT TO DO ABOUT UNKNOWN'''
cleaned_smoking = cleaned_gender.copy()
cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'formerly smoked',
                    'smoking_status'] = 1
cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'smokes',
                    'smoking_status'] = 1
cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'never smoked',
                    'smoking_status'] = 0
print(cleaned_smoking)
''' for BMI, according to the CDC
If your BMI is less than 18.5, it falls within the underweight range.
If your BMI is 18.5 to <25, it falls within the healthy weight range.
If your BMI is 25.0 to <30, it falls within the overweight range.
If your BMI is 30.0 or higher, it falls within the obesity range.
'''
# for now, convert BMI into categories
# then determine if we can convert to binary
cleaned_BMI = cleaned_smoking.copy()
cleaned_BMI['BMI_cat'] = np.where(cleaned_BMI['bmi'] < 18.5,
                                  'underweight',
                                  np.where((cleaned_BMI['bmi'] >= 18.5) & (cleaned_BMI['bmi'] <= 24.9),
                                           'healthy weight',
                                           np.where((cleaned_BMI['bmi'] >= 25) & (cleaned_BMI['bmi'] <= 29.9),
                                                    'overweight',
                                                    'obese')))


#print(cleaned_BMI)

# convert avg_glucose_level into binary
# 99 or below is 0, 100+ is 1
# source: cdc gov

cleaned_glucose = cleaned_BMI.copy()
cleaned_glucose['cleaned_avg_glucose'] = np.where((cleaned_glucose['avg_glucose_level'] <= 99),
                                                  0, np.where((cleaned_glucose['avg_glucose_level'] >= 100),
                                                              1, cleaned_glucose['avg_glucose_level'])).astype(int)

#print(cleaned_glucose)

# convert age into binary
# less than 65 is 0
# 65+ is 1

cleaned_age = cleaned_glucose.copy()
cleaned_age['cleaned_age'] = np.where((cleaned_age['age'] <= 64),
                                              0, np.where((cleaned_age['age'] >= 65),
                                                          1, cleaned_age['age'])).astype(int)
print(cleaned_age)


def create_risk_factor_df(stroke_data: str) -> pd.DataFrame:
    '''
    Cleans the stroke data file. Converts all columns of interest to binary.
    To convert data in gender column. Male = 1 and Female = 0
    we do not worry about other genders because they are not
    included in the dataset. To convert data in smoking_status column:
    has smoked = 1.
    Never smoked = 0. To convert BMI, according to the CDC
    If your BMI is less than 18.5, it falls within the underweight range.
    If your BMI is 18.5 to <25, it falls within the healthy weight range.
    If your BMI is 25.0 to <30, it falls within the overweight range.
    If your BMI is 30.0 or higher, it falls within the obesity range.
    To convert avg_glucose_level into binary 99 or below is 0, 100+ is 1.
    To convert age into binary less than 65 is 0 , 65+ is 1
    To convert residence_type to binary 1 for urban 0 for rural.
    To convert ever_married to binary 1 for married 0 for not married
    '''
    unclean_data = pd.read_csv(stroke_data)
    cleaned_gender = unclean_data.copy()
    cleaned_gender.loc[cleaned_gender["gender"] == "Male", "gender"] = 1
    cleaned_gender.loc[cleaned_gender["gender"] == "Female", "gender"] = 0
    cleaned_smoking = cleaned_gender.copy()
    cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'formerly smoked',
                        'smoking_status'] = 1
    cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'smokes',
                        'smoking_status'] = 1
    cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'never smoked',
                        'smoking_status'] = 0
    cleaned_BMI = cleaned_smoking.copy()
    cleaned_BMI['low_BMI'] = np.where(cleaned_BMI['bmi'] < 18.5, 1, 0)
    cleaned_BMI['high_BMI'] = np.where((cleaned_BMI['bmi'] >= 18.5) &
                                       (cleaned_BMI['bmi'] <= 24.9), 0,
                                np.where((cleaned_BMI['bmi'] >= 25) &
                                        (cleaned_BMI['bmi'] <= 29.9), 1, 0))
    cleaned_glucose = cleaned_BMI.copy()
    cleaned_glucose['high_glucose'] = np.where((cleaned_glucose['avg_glucose_level']
                                                <= 99), 0,
                                                np.where((cleaned_glucose['avg_glucose_level']
                                                          >= 100),
                                                         1, cleaned_glucose['avg_glucose_level'])).astype(int)
    cleaned_age = cleaned_glucose.copy()
    cleaned_age['over_65'] = np.where((cleaned_age['age'] <= 64),
                                                0, np.where((cleaned_age['age'] >= 65),
                                                            1, cleaned_age['age'])).astype(int)
    cleaned_residence = cleaned_age.copy()
    cleaned_residence.loc[cleaned_residence["Residence_type"]
                          == "Urban", "residence"] = 1
    cleaned_residence.loc[cleaned_residence["Residence_type"]
                          == "Rural", "residence"] = 0
    cleaned_married = cleaned_residence.copy()
    cleaned_married.loc[cleaned_married["ever_married"]
                      == "Yes", "married"] = 1
    cleaned_married.loc[cleaned_married["ever_married"]
                      == "No", "married"] = 0
    final_df = cleaned_married[['hypertension', 'heart_disease', 'high_glucose', 'low_BMI',
                        'high_BMI', 'gender', 'over_65', 'married',
                        'residence', 'smoking_status', 'stroke']]
    return final_df


# Calculate the correlation matrix
corr_matrix = cleaned_age.corr()

# Extract the correlation coefficients for stroke
stroke_corr = corr_matrix["stroke"]

# Sort the correlation coefficients in descending order
stroke_corr_sorted = stroke_corr.abs().sort_values(ascending=False)

# Print the sorted correlation coefficients
print(stroke_corr_sorted)

def create_risk_factor_df(stroke_data: str) -> pd.DataFrame:
    '''
    Cleans the stroke data file. Converts all columns of interest to binary.
    To convert data in gender column. Male = 1 and Female = 0
    we do not worry about other genders because they are not
    included in the dataset. To convert data in smoking_status column:
    has smoked = 1.
    Never smoked = 0. To convert BMI, according to the CDC
    If your BMI is less than 18.5, it falls within the underweight range.
    If your BMI is 18.5 to <25, it falls within the healthy weight range.
    If your BMI is 25.0 to <30, it falls within the overweight range.
    If your BMI is 30.0 or higher, it falls within the obesity range.
    To convert avg_glucose_level into binary 99 or below is 0, 100+ is 1.
    To convert age into binary less than 65 is 0 , 65+ is 1
    To convert residence_type to binary 1 for urban 0 for rural.
    To convert ever_married to binary 1 for married 0 for not married
    '''
    unclean_data = pd.read_csv(stroke_data)
    cleaned_gender = unclean_data.copy()
    cleaned_gender.loc[cleaned_gender["gender"] == "Male", "gender"] = 1
    cleaned_gender.loc[cleaned_gender["gender"] == "Female", "gender"] = 0
    cleaned_smoking = cleaned_gender.copy()
    cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'formerly smoked',
                        'smoking_status'] = 1
    cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'smokes',
                        'smoking_status'] = 1
    cleaned_smoking.loc[cleaned_smoking['smoking_status'] == 'never smoked',
                        'smoking_status'] = 0
    cleaned_BMI = cleaned_smoking.copy()
    cleaned_BMI['low_BMI'] = np.where(cleaned_BMI['bmi'] < 18.5, 1, 0)
    cleaned_BMI['high_BMI'] = np.where((cleaned_BMI['bmi'] >= 18.5) &
                                       (cleaned_BMI['bmi'] <= 24.9), 0, 
                                np.where((cleaned_BMI['bmi'] >= 25) & 
                                        (cleaned_BMI['bmi'] <= 29.9), 1, 0))
    cleaned_glucose = cleaned_BMI.copy()
    cleaned_glucose['high_glucose'] = np.where((cleaned_glucose['avg_glucose_level']
                                                <= 99),0,
                                                np.where((cleaned_glucose['avg_glucose_level']
                                                          >= 100),
                                                         1, cleaned_glucose['avg_glucose_level'])).astype(int)
    cleaned_age = cleaned_glucose.copy()
    cleaned_age['over_65'] = np.where((cleaned_age['age'] <= 64),
                                                0, np.where((cleaned_age['age'] >= 65),
                                                            1, cleaned_age['age'])).astype(int)
    cleaned_residence = cleaned_age.copy()
    cleaned_residence.loc[cleaned_residence["Residence_type"]
                          == "Urban", "residence"] = 1
    cleaned_residence.loc[cleaned_residence["Residence_type"]
                          == "Rural", "residence"] = 0
    cleaned_married = cleaned_residence.copy()
    cleaned_married.loc[cleaned_married["ever_married"]
                      == "Yes", "married"] = 1
    cleaned_married.loc[cleaned_married["ever_married"]
                      == "No", "married"] = 0
    final_df = cleaned_married[['hypertension', 'heart_disease', 'high_glucose', 'low_BMI',
                        'high_BMI','gender', 'over_65', 'married',
                        'residence', 'smoking_status', 'stroke']]
    return final_df

def create_shapefile_for_bubble_map(shapefile: str, hypertension: str, obesity: str, 
                                    diabetes: str,abbr: str, stroke:str) -> gpd.GeoDataFrame:
    '''
    Question 2:

    Since we see Heart Disease has highest correlation with stroke
    map different heart disease/stroke risk factors
    Find datasets
    on hypertension, geography, state, and population size
    high BMI (obesity), geography,state, and population size
    on High glucose (diabetes), geography,state and population size
    Combine on state
    Find dataset
    on map of strokes
    Plotly bubble map to layer map of strokes over map of each health risk factor
    '''


    # create shapefile for geometry and state name
    us_shapefile = gpd.read_file(shapefile)
    us_shapefile = us_shapefile.rename(columns={'NAME': 'State'})


    # hypertension by state
    hypertension_state = pd.read_excel(hypertension, engine='openpyxl')
    hypertension_state = hypertension_state[hypertension_state['State']
                                            != 'Virgin Islands']

    # Obesity (high-BMI by state)
    obesity_state = pd.read_csv(obesity)
    obesity_state = obesity_state[obesity_state['State'] != 'Virgin Islands']
    obesity_state = obesity_state[["State", "Prevalence"]]
    obesity_state = obesity_state.rename(
        columns={'Prevalence': 'Obesity_prev_perc'})

    # high-glucose (Diabetes by state)
    raw_diabetes_state = pd.read_csv(diabetes)
    state_abbr = pd.read_csv(abbr)
    state_abbr = state_abbr[["code", "state"]]
    raw_diabetes_state = raw_diabetes_state.groupby(
        'state_abbr', as_index=False).mean()
    diabetes_state = state_abbr.merge(
        raw_diabetes_state, left_on='code', right_on='state_abbr', how='outer')
    diabetes_state = diabetes_state[diabetes_state['state'] != 'Virgin Islands']
    diabetes_state = diabetes_state[["state", "value"]]
    diabetes_state = diabetes_state.rename(
        columns={'value': 'Diabetes_prev_perc', 'state': 'State'})

    # stroke mortality by state
    stroke_mortality_df = pd.read_csv(stroke)
    year_2020 = stroke_mortality_df["YEAR"] == 2020
    stroke_mortality_df = stroke_mortality_df[year_2020]
    stroke_mortality_df = stroke_mortality_df[stroke_mortality_df['STATE']
                                            != 'Virgin Islands']
    stroke_mortality_df = stroke_mortality_df[["STATE", "RATE"]]
    stroke_mortality_df = state_abbr.merge(
        stroke_mortality_df, left_on='code', right_on='STATE', how='outer')
    stroke_mortality_df = stroke_mortality_df[stroke_mortality_df['state']
                                            != 'Virgin Islands']
    stroke_mortality_df = stroke_mortality_df[["state", "RATE"]]
    stroke_mortality_df = stroke_mortality_df.rename(
        columns={'RATE': 'stroke_mortality_rate', 'state': 'State'})

    # combine datasets
    dataframes_to_merge = [hypertension_state,
                        obesity_state, diabetes_state, stroke_mortality_df]
    merged_risk_factors = reduce(lambda left, right: pd.merge(
        left, right, on=['State'], how='outer'), dataframes_to_merge)


    excluded_states = ['United States Virgin Islands',
                       'Commonwealth of the Northern Mariana Islands', 'Guam', 'American Samoa', 'District of Columbia', 'Puerto Rico']  # list of state codes to exclude
    us_shapefile = us_shapefile[~us_shapefile['State'].isin(excluded_states)]
    merged_risk_factors = merged_risk_factors[~merged_risk_factors['State'].isin(
        excluded_states)]

    risk_factors_and_stroke_df = us_shapefile.merge(
        merged_risk_factors, left_on='State', right_on='State', how='inner')
    
    return risk_factors_and_stroke_df


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
