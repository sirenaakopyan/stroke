'''
This file contains functions that clean up the csv dataset
and attempt to find the top 3 demographic and health risk
factors for stroke.
'''
import pandas as pd
import numpy as np

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

# Calculate the correlation matrix
corr_matrix = cleaned_age.corr()

# Extract the correlation coefficients for stroke
stroke_corr = corr_matrix["stroke"]

# Sort the correlation coefficients in descending order
stroke_corr_sorted = stroke_corr.abs().sort_values(ascending=False)

# Print the sorted correlation coefficients
print(stroke_corr_sorted)

# result
# cleaned_age            0.247136
# age                    0.245257
# heart_disease          0.134914
# avg_glucose_level      0.131945
# hypertension           0.127904
# bmi                    0.042374
# cleaned_avg_glucose    0.012812

# => highest correlate factor is age & second is heart_disease weiufgweskuhd tesetig git commit