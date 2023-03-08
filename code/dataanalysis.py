import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go



def find_risk_factor_correlation(risk_factor_df: pd.DataFrame) -> float:
    '''Calculate the correlation matrix.
    Sort the correlation coefficients in descending order.
    Return the sorted correlation coefficients.
    '''
    corr_matrix = risk_factor_df.corr()
    stroke_corr = corr_matrix["stroke"]
    stroke_corr_sorted = stroke_corr.abs().sort_values(ascending=False)
    return stroke_corr_sorted

def risk_factor_df_ML(dataframe: str) -> pd.DataFrame:
    correlations = find_risk_factor_correlation(dataframe)
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
    dataframe3 = dataframe2.copy()
    dataframe3.loc[:, 'row_corr'] = dataframe3.sum(axis=1)/ (hypertension + heart_disease + high_glucose + low_bmi
                                                             + high_BMI + over_65 + married + residence)
    dataframe3 = dataframe3[['row_corr']]
    mL_df = pd.merge(dataframe, dataframe3, left_index=True, right_index=True)
    return mL_df
    
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

    # re-project the geometry column to a projected CRS before calculating the centroid coordinates
    #map_data = map_data.to_crs("EPSG:3395")

    # Create a choropleth map of stroke mortality by state
    fig = px.choropleth(
    map_data, 
    geojson=us_map.geometry.__geo_interface__,
        locations='STATEFP',
    color='stroke_mortality_rate',
    color_continuous_scale='Blues',
    range_color=(0, map_data['stroke_mortality_rate'].max()),
    scope='usa',
    hover_data=['State', 'stroke_mortality_rate'], # Add 'stroke_mortality_rate' to the list
    labels={'stroke_mortality_rate': 'Stroke Mortality Rate'}
)


    # # Use plotly to create a scatter_geo trace for hypertension by state
    # fig.add_trace(
    #     go.Scattergeo(
    #         lon=map_data["INTPTLON"],
    #         lat=map_data["INTPTLAT"],
    #         text=map_data["STUSPS"],
    #         marker=dict(
    #             size=map_data.filter(
    #                 like='Percent_with_hypertension').iloc[:, 0]*20,
    #             color="blue",
    #             opacity=0.5,
    #             sizemode="diameter",
    #             sizemin=4
    #         ),
    #         hoverinfo="text"
    #     )
    # )

    fig.update_layout(
        title={
            "text": "Stroke Mortality and Hypertension by State",
            "y":0.98,
            "x":0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        geo_scope="usa",
        height=600
    )

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
    hypertension = pd.read_excel("datasets/hypertension_by_state.xlsx")
    print("\n================================")

    print(hypertension.columns)
    # us_map = gpd.read_file("datasets/tl_2017_us_state/tl_2017_us_state.shp")
    #print(us_map.columns)
    # hypertension = pd.read_excel("datasets/hypertension_by_state.xlsx")
    #print(hypertension.columns)
    print(find_risk_factor_correlation(risk_factor_data))
    print(risk_factor_df_ML(risk_factor_data))
    map_risk_factors(map_data)



if __name__ == '__main__':
    main()
