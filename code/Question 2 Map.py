import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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
                        labels={
                            'stroke_mortality_rate': 'Stroke Mortality Rate', 'STUSPS': 'State ID'}
                        )

    fig.update_layout(
        title={
            "text": "Stroke Mortality and Hypertension by State",
            "xanchor": "center",
            "yanchor": "top"
        },
        title_font_family="Times New Roman",
        title_font_size=26,
        title_font_color="black",
        title_x=.45
    )

    fig = px.scatter_geo(map_data, locations="STUSPS", color="Percent_with_hypertension",
                         hover_name="Percent_with_hypertension", size="Percent_with_hypertension",
                         projection="usa")

    fig.show()


def main():
    map_data = datacleanup.create_shapefile_for_bubble_map(
        "datasets/tl_2017_us_state/tl_2017_us_state.shp",
        "datasets/hypertension_by_state.xlsx",
        "datasets/Obesity_by_state.csv",
        "datasets/Diabetes_by_state.csv",
        "datasets/State_code_to_name.csv",
        "datasets/stroke_mortality_state.csv")
    map_risk_factors(map_data)


if __name__ == '__main__':
    main()
