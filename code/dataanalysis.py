import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import operator

def normalize_data(df):
    shuffled_data = df.sample(frac=1,random_state=4)
    stroke_df = df.loc[df['stroke'] == 1]
    non_stroke_df = df.loc[df['stroke'] == 0].sample(n=1000, random_state=101)
    normalized_stroke = pd.concat([stroke_df, non_stroke_df])
    return normalized_stroke


def cramers_V(var1:str, var2:str) -> float:
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))


def find_correlations(df: pd.DataFrame):
    corr_dict = {}
    for col in df.columns:
        corr = (col, cramers_V(df[col], df['stroke']))
        corr_dict[col] = corr[1]
    return corr_dict


def sorted_correlations(corr_dict):
    sorted_d = dict( sorted(corr_dict.items(), key=operator.itemgetter(1),reverse=True))
    return sorted_d


def find_risk_factor_correlation(risk_factor_df: pd.DataFrame) -> float:
    '''Calculate the correlation matrix.
    Sort the correlation coefficients in descending order.
    Return the sorted correlation coefficients.
    '''
    corr_matrix = risk_factor_df.corr()
    stroke_corr = corr_matrix["stroke"]
    stroke_corr_sorted = stroke_corr.abs().sort_values(ascending=False)
    return stroke_corr_sorted
'''
def pair_visualization(df):

    fig = plt.figure(figsize=(20,15),dpi=100)
    sns.pairplot(data=df,hue='stroke',size=2,palette='OrRd')
    plt.savefig('kk.png',  bbox_inches='tight')
'''

def comparison_bar_charts(risk_factor_df: pd. DataFrame) -> None:
    # bar chart 1: stroke vs age
    over_65_stroke = sns.countplot(x='stroke', hue = 'over_65', data = risk_factor_df, palette = "Set1")
    over_65_stroke.set_xticklabels(["No", "Yes"])
    over_65_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Age', labels=['Under 65', 'Over 65'])
    plt.title("Age and Stroke")
    plt.savefig('code/question1images/over_65.png',  bbox_inches='tight')
    # plt.show()
    
    # bar chart 2: stroke vs smoking
    smoking_status_stroke = sns.countplot(x='stroke', hue = 'smoking_status', data = risk_factor_df, palette = "Set1")
    smoking_status_stroke.set_xticklabels(["No", "Yes"])
    smoking_status_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Smoking Status', labels=['Not a Smoker', 'Smoker', 'Unknown'])
    plt.title("Smoking Status and Stroke")
    plt.savefig('code/question1images/smoking_status.png',  bbox_inches='tight')
    # plt.show()

    # bar chart 3: stroke vs gender
    gender_stroke = sns.countplot(x='stroke', hue = 'gender', data = risk_factor_df, palette = "Set1")
    gender_stroke.set_xticklabels(["No", "Yes"])
    gender_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Gender', labels=['Female', 'Male'])
    plt.title("Gender and Stroke")
    plt.savefig('code/question1images/gender.png',  bbox_inches='tight')
    # plt.show()

    # bar chart 4: stroke vs heart disease
    heart_disease_stroke = sns.countplot(x='stroke', hue = 'heart_disease', data = risk_factor_df, palette = "Set1")
    heart_disease_stroke.set_xticklabels(["No", "Yes"])
    heart_disease_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Status', labels=['did not have heart disease', 'had heart disease'])
    plt.title("Heart Disease and Stroke")
    plt.savefig('code/question1images/heart_disease.png',  bbox_inches='tight')
    # plt.show()

    # bar chart 5: stroke vs hypertension
    hypertension_and_stroke = sns.countplot(x='stroke', hue = 'hypertension', data = risk_factor_df, palette = "Set1")
    hypertension_and_stroke.set_xticklabels(["No", "Yes"])
    hypertension_and_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Status', labels=['did not have hypertension', 'had hypertension'])
    plt.title("Hypertension and Stroke")
    plt.savefig('code/question1images/hypertension.png',  bbox_inches='tight')
    # plt.show()

    # bar chart 6: stroke vs married
    marital_status_stroke = sns.countplot(x='stroke', hue = 'married', data = risk_factor_df, palette = "Set1")
    marital_status_stroke.set_xticklabels(["No", "Yes"])
    marital_status_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Marital Status', labels=['not married', 'married'])
    plt.title("Marital Status and Stroke")
    plt.savefig('code/question1images/Marital_status.png',  bbox_inches='tight')
    # plt.show()
    # bar chart 7: stroke vs high glucose
    high_glucose_stroke = sns.countplot(x='stroke', hue = 'high_glucose', data = risk_factor_df, palette = "Set1")
    high_glucose_stroke.set_xticklabels(["No", "Yes"])
    high_glucose_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Status', labels=['normal glucose', 'high glucose'])
    plt.title("Glucose level and Stroke")
    plt.savefig('code/question1images/glucose.png',  bbox_inches='tight')
    # plt.show()

    # bar chart 8: stroke vs high BMI
    high_bmi_stroke = sns.countplot(x='stroke', hue = 'high_BMI', data = risk_factor_df, palette = "Set1")
    high_bmi_stroke.set_xticklabels(["No", "Yes"])
    high_bmi_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Status', labels=['normal BMI', 'high BMI'])
    plt.title("High BMI and Stroke")
    plt.savefig('code/question1images/high_bmi.png',  bbox_inches='tight')
    # plt.show()

    # bar chart 9: stroke vs low BMI
    low_bmi_stroke = sns.countplot(x='stroke', hue = 'low_BMI', data = risk_factor_df, palette = "Set1")
    low_bmi_stroke.set_xticklabels(["No", "Yes"])
    low_bmi_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='Status', labels=['normal BMI', 'low BMI'])
    plt.title("low BMI and Stroke")
    plt.savefig('code/question1images/low_bmi.png',  bbox_inches='tight')
    #plt.show()

    # bar chart 10: stroke vs residence
    residence_stroke = sns.countplot(x='stroke', hue = 'residence', data = risk_factor_df, palette = "Set1")
    residence_stroke.set_xticklabels(["No", "Yes"])
    residence_stroke.set(xlabel = 'Had Stroke', ylabel = 'Count')
    plt.legend(title='residence type', labels=['rural', 'urban'])
    plt.title("Residence Type and Stroke")
    plt.savefig('code/question1images/residence_type.png',  bbox_inches='tight')
    # plt.show()

    # bar chart plotting when stroke = 1 


'''
def visualization_correlation_matrix(df):
    
    # feature log transformations 
    df['bmi_cat'] = pd.cut(df['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
    df['age_cat'] = pd.cut(df['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
    df['glucose_cat'] = pd.cut(df['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])
    df_copy = df.copy()
    df_copy['age'] = df_copy['age'].apply(lambda x: np.log(x+10)*3)
    df_copy['avg_glucose_level'] = df_copy['avg_glucose_level'].apply(lambda x: np.log(x+10)*2)
    df_copy['bmi'] = df_copy['bmi'].apply(lambda x: np.log(x+10)*2)

    ## label encoding of ordinal categorical features
    for col in df_copy.columns:
        df_copy[col] = le.fit_transform(df_copy[col])

    cols = df_copy.columns
    ## normalizing with standard scaler of numerical features
    df_copy[cols] = ss.fit_transform(df_copy[cols])


    # correlation map for all the features
    df_corr = df_copy.drop(columns = ['id']).corr()
    # mask = np.triu(np.ones_like(df_corr, dtype=np.bool))

    fig, ax = plt.subplots(figsize = (8,8))
    fig.patch.set_facecolor('#f6f5f5')
    ax.set_facecolor('#f6f5f5')

    # mask = mask[1:, :-1]
    # corr = df_corr.iloc[1:,:-1].copy()

    # plot heatmap
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Blues",
               vmin=-0.15, vmax=0.5, ax = ax, cbar = True,
               linewidth = 1,linecolor = '#f6f5f5', square = True,annot_kws = {'font':'serif', 'size':10, 'color':'black'} )
    # yticks
    ax.tick_params(axis = 'y', rotation=0)
    xticks = ['Gender', 'Age','Hyper tension', 'Heart Disease', 'Marriage', 'Work', 'Residence', 'Glucose Level', 'BMI', 'Smoking Status','Stroke','BMI Cat','Age Cat']
    yticks = ['Gender', 'Age','Hyper tension', 'Heart Disease', 'Marriage', 'Work', 'Residence', 'Glucose Level', 'BMI', 'Smoking Status','Stroke','BMI Cat','Age Cat']
    # ax.set_xticklabels(xticks, {'font':'serif', 'size':10, 'weight':'bold'},rotation = 90, alpha = 0.9)
    # ax.set_yticklabels(yticks, {'font':'serif', 'size':10, 'weight':'bold'}, rotation = 0, alpha = 0.9)
    fig.show()
    plt.savefig('foo.png',  bbox_inches='tight')
'''

'''
def map_risk_factors(map_data: pd.DataFrame):
    """
    Display a bubble map of the top risk factors 
    across the country.
    """

    fig = px.choropleth(map_data,
                        locations='STUSPS',
                        locationmode="USA-states",
                        scope="usa",
                        color='stroke_mortality_rate',
                        color_continuous_scale="Viridis_r",
                        hover_data=['State', 'stroke_mortality_rate'],
                        labels={'stroke_mortality_rate': 'Stroke Mortality Rate', 'STUSPS': 'State ID'}
                        )
    
    fig.update_layout(
            title={
                "text": "Stroke Mortality and Hypertension by State",
                "y":0.98,
                "x":0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
        title_font_family="Times New Roman",
        title_font_size=26,
        title_font_color="black",
        title_x=.45
    )

    # fig = px.scatter_geo(map_data, locations="STUSPS", color="Percent_with_hypertension",
    #                      hover_name="Percent_with_hypertension", size="Percent_with_hypertension",
    #                      projection="usa")

    fig.show()
'''

def main():
    risk_factor_data = datacleanup.create_risk_factor_df(
        'datasets/stroke_data_1.csv')
    map_data = datacleanup.create_shapefile_for_bubble_map(
        "datasets/tl_2017_us_state/tl_2017_us_state.shp",
        "datasets/hypertension_by_state.xlsx",
        "datasets/Obesity_by_state.csv",
        "datasets/Diabetes_by_state.csv",
        "datasets/State_code_to_name.csv",
        "datasets/stroke_mortality_state.csv")  
    # print("\n================================")       
    # print(map_data.columns)
    # print("\n================================")

    # us_map = gpd.read_file("datasets/tl_2017_us_state/tl_2017_us_state.shp")
    # print(us_map.columns)
    #hypertension = pd.read_excel("datasets/hypertension_by_state.xlsx", engine='openpyxl')
    # print("\n================================")

    # df = pd.read_csv('datasets/stroke_data_1.csv')
    # visualization_correlation_matrix(df)
    # pair_visualization(df)
    normalized_risk_factor = normalize_data(risk_factor_data)

    correlations = find_correlations(normalized_risk_factor)
    print(sorted_correlations(correlations))
    # print(find_risk_factor_correlation(risk_factor_data))
    comparison_bar_charts(normalized_risk_factor)

    # map_risk_factors(map_data)
    # print("\n================================")
   
    


if __name__ == '__main__':
    main()
