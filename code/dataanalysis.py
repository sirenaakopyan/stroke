import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


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


    
def map_risk_factors(map_data: pd.DataFrame):
    """
    Display a bubble map of the top risk factors 
    across the country.
    """
    # Load the shapefile into a geopandas dataframe
    us_map = gpd.read_file("datasets/tl_2017_us_state/tl_2017_us_state.shp")

    # Load the hypertension data into a pandas dataframe
    hypertension = pd.read_excel("datasets/hypertension_by_state.xlsx")


    # Merge the geopandas dataframe with the hypertension dataframe
    map_data = map_data.merge(hypertension, on="State")
    map_data = map_data.reset_index()

    print(map_data)
    print(map_data.columns)

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
        title_x=.45,
    )

    # fig = px.scatter_geo(map_data, locations="STUSPS", color="Percent_with_hypertension",
    #                      hover_name="Percent_with_hypertension", size="Percent_with_hypertension",
    #                      projection="usa")

    fig.show()


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

    us_map = gpd.read_file("datasets/tl_2017_us_state/tl_2017_us_state.shp")
    print(us_map.columns)
    hypertension = pd.read_excel("datasets/hypertension_by_state.xlsx", engine='openpyxl')
    print("\n================================")

    df = pd.read_csv('datasets/stroke_data_1.csv')
    visualization_correlation_matrix(df)
    pair_visualization(df)
    print(find_risk_factor_correlation(risk_factor_data))

    map_risk_factors(map_data)
    print("\n================================")
   
    


if __name__ == '__main__':
    main()
