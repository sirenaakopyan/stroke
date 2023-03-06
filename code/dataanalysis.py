import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd

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
    
def map_risk_factors(dataframe: str):
    """
    Display a bubble map of the top risk factors 
    across the country.
    """
    # # load the state codes and names
    # state_codes = pd.read_csv("datasets/State_code_to_name.csv")

    # # merge the data with the state codes
    # merged_data = pd.merge(dataframe, state_codes, on="state_code")

    # # create a scatter_geo trace for hypertension by state
    # fig = px.scatter_geo(merged_data, locations="state_code",
    #                      locationmode="USA-states", color="state_name",
    #                      size="hypertension", hover_name="state_name",
    #                      hover_data=["hypertension"])

    # # load the stroke mortality data by state
    # stroke_data = pd.read_csv("datasets/stroke_mortality_state.csv")
    # # merge with the state codes
    # merged_stroke_data = pd.merge(stroke_data, state_codes, on="state_code")

    # # add a choropleth map of stroke mortality by state
    # fig.add_choropleth(locations=merged_stroke_data["state_code"],
    #                    z=merged_stroke_data["stroke_mortality"],
    #                    locationmode="USA-states",
    #                    colorscale="Reds", colorbar_title="Stroke Mortality",
    #                    geojson=counties_json)

    # fig.update_layout(title="Hypertension and Stroke Mortality by State",
    #                   geo_scope="usa")
    pass


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
    us_map = gpd.read_file("datasets/tl_2017_us_state/tl_2017_us_state.shp")
    print(us_map.columns)
    print(find_risk_factor_correlation(risk_factor_data))
    print(risk_factor_df_ML(risk_factor_data))
    map_risk_factors(map_data)



if __name__ == '__main__':
    main()
