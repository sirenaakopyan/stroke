'''
this file contains functions that clean up the csv dataset
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
cleaned_BMI.loc[cleaned_BMI['bmi'] <= 18.4, 'BMI_cat'] = 'underweight'
cleaned_BMI['BMI_cat'] = np.where((cleaned_BMI['bmi'] >= 18.5)
                                  & (cleaned_BMI['bmi'] <= 24.9),
                                  'healthy weight', cleaned_BMI['bmi'])
cleaned_BMI['BMI_cat'] = np.where((cleaned_BMI['bmi'] >= 25)
                                  & (cleaned_BMI['bmi'] <= 29.9),
                                  'overweight', cleaned_BMI['bmi'])
cleaned_BMI.loc[cleaned_BMI['bmi'] >= 30, 'BMI_cat'] = 'obese'
