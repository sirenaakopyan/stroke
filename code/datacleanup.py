'''
this file contains functions that clean up the csv dataset
and attempt to find the top 3 demographic and health risk
factors for stroke.
'''
import pandas as pd
import numpy as np
import geopandas as gpd
from functools import reduce

# Question 1: What does the data indicate are the risk factors
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
cleaned_BMI['low_BMI'] = np.where(cleaned_BMI['bmi'] < 18.5, 1, 0)
cleaned_BMI['high_BMI'] = np.where((cleaned_BMI['bmi'] >= 18.5) & (
    cleaned_BMI['bmi'] <= 24.9), 0, np.where((cleaned_BMI['bmi'] >= 25)
                                             & (cleaned_BMI['bmi'] <= 29.9), 1, 0))


#print(cleaned_BMI)

# convert avg_glucose_level into binary
# 99 or below is 0, 100+ is 1
# source: cdc gov

cleaned_glucose = cleaned_BMI.copy()
cleaned_glucose['high_glucose'] = np.where((cleaned_glucose['avg_glucose_level'] <= 99),
                                                  0, np.where((cleaned_glucose['avg_glucose_level'] >= 100),
                                                              1, cleaned_glucose['avg_glucose_level'])).astype(int)

#print(cleaned_glucose)

# convert age into binary
# less than 65 is 0
# 65+ is 1

cleaned_age = cleaned_glucose.copy()
cleaned_age['over_65'] = np.where((cleaned_age['age'] <= 64),
                                              0, np.where((cleaned_age['age'] >= 65),
                                                          1, cleaned_age['age'])).astype(int)
# print(cleaned_age)
# print(cleaned_age.columns)

# converting residence_type to binary
# 1 for urban
# 0 for rural
cleaned_residence = cleaned_age.copy()
cleaned_residence.loc[cleaned_residence["Residence_type"]
                      == "Urban", "residence"] = 1
cleaned_residence.loc[cleaned_residence["Residence_type"]
                      == "Rural", "residence"] = 0
# print(cleaned_residence)

# converting ever_married to binary
# 1 for married
# 0 for not married
cleaned_married = cleaned_residence.copy()
cleaned_married.loc[cleaned_married["ever_married"]
                      == "Yes", "married"] = 1
cleaned_married.loc[cleaned_married["ever_married"]
                      == "No", "married"] = 0
# print(cleaned_married.columns)

# converting work type to binary


final_df = cleaned_married[['hypertension', 'heart_disease', 'high_glucose', 'low_BMI',
                        'high_BMI','gender', 'over_65', 'married', 'work_type',
                        'residence', 'smoking_status', 'stroke']]
print (final_df)
# Calculate the correlation matrix
corr_matrix = final_df.corr()

# Extract the correlation coefficients for stroke
stroke_corr = corr_matrix["stroke"]

# Sort the correlation coefficients in descending order
stroke_corr_sorted = stroke_corr.abs().sort_values(ascending=False)

# Print the sorted correlation coefficients
print(stroke_corr_sorted)

# result
# stroke           1.000000
# over_65          0.247136
# heart_disease    0.134914
# hypertension     0.127904
# married          0.108340
# low_BMI          0.056477
# residence        0.015458
# high_BMI         0.012900
# high_glucose     0.012812

# => highest correlation factor is age & second is heart_disease

# plotly plots for occurrences of 


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
us_shapefile = gpd.read_file("datasets/tl_2017_us_state/tl_2017_us_state.shp")
us_shapefile = us_shapefile[["NAME", "geometry"]]
us_shapefile = us_shapefile.rename(columns={'NAME': 'State'})
print(us_shapefile)

# hypertension by state
hypertension_state = pd.read_excel("datasets/hypertension_by_state.xlsx", engine='openpyxl')
print(hypertension_state)

# Obesity (high-BMI by state)
obesity_state = pd.read_csv("datasets/Obesity_by_state.csv")
obesity_state = obesity_state[["State", "Prevalence"]]
obesity_state = obesity_state.rename(columns={'Prevalence': 'Obesity_prev_perc'})
print(obesity_state)

# high-glucose (Diabetes by state)
raw_diabetes_state = pd.read_csv("datasets/Diabetes_by_state.csv")
state_abbr = pd.read_csv("datasets/State_code_to_name.csv")
state_abbr = state_abbr[["code", "state"]]
raw_diabetes_state = raw_diabetes_state.groupby('state_abbr', as_index=False).mean()
diabetes_state = state_abbr.merge(raw_diabetes_state, left_on='code',
                                   right_on='state_abbr', how='outer')
diabetes_state = diabetes_state[["state", "value"]]
diabetes_state = diabetes_state.rename(columns={'value': 'Diabetes_prev_perc', 'state': 'State'})
print(diabetes_state)

# stroke mortality by state
stroke_mortality_df = pd.read_csv("datasets/stroke_mortality_state.csv")
year_2020 = stroke_mortality_df["YEAR"] == 2020
stroke_mortality_df = stroke_mortality_df[year_2020]
stroke_mortality_df = stroke_mortality_df[["STATE", "RATE"]]
stroke_mortality_df = state_abbr.merge(stroke_mortality_df, left_on='code',
                                   right_on='STATE', how='outer')
stroke_mortality_df = stroke_mortality_df[["state", "RATE"]]
stroke_mortality_df = stroke_mortality_df.rename(columns={'RATE': 'stroke_mortality_rate', 'state': 'State'})
print(stroke_mortality_df)

# combine datasets
dataframes_to_merge = [hypertension_state, obesity_state, diabetes_state, stroke_mortality_df]
merged_risk_factors = reduce(lambda  left,right: pd.merge(left,right,on=['State'],
                                            how='outer'), dataframes_to_merge)
print(merged_risk_factors)
 # combine with shapefile
risk_factors_and_stroke_df = us_shapefile.merge(merged_risk_factors, left_on='State',
                                    right_on='State', how='inner')
print(risk_factors_and_stroke_df.columns)
