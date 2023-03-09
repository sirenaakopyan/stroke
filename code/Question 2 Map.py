import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.subplots as sp


def map_risk_factors(map_data: pd.DataFrame):
    """
    Display a bubble map of the top risk factors 
    across the country.
    """

    # print(map_data)
    # print(map_data.columns)

    fig = go.Figure()

    hyp_map_data = map_data.dropna(subset=['Percent_with_hypertension'])

    fig.add_trace(go.Scattergeo(
        locations=hyp_map_data['STUSPS'],
        locationmode='USA-states',

        # Longitude of center of the United States
        lon=[-98.5] * len(hyp_map_data),
        # Latitude of center of the United States
        lat=[39.8] * len(hyp_map_data),
        marker=dict(
            size=hyp_map_data['Percent_with_hypertension'],
            sizemode='diameter',
            sizeref=10000000000 * hyp_map_data['Percent_with_hypertension'].max() / (7 ** 14),
            color=hyp_map_data['Percent_with_hypertension'],
            # line=dict(
            #     width=1,
            #     color='grey',
            # )
            colorscale='greys'
        ),
        name='Percent with Hypertension',
        text=hyp_map_data['Percent_with_hypertension'].apply(
            lambda x: f'{x:.2f}%'),
        hovertemplate="%{text} of population<br>" +
        "<extra></extra>"
    ))

    fig.add_trace(go.Choropleth(
        locations=map_data['STUSPS'],
        z=map_data['stroke_mortality_rate'],
        locationmode="USA-states",
        colorscale="Viridis_r",
        zmax=map_data['stroke_mortality_rate'].max(),
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar_title="Stroke Mortality Rate",
        hovertemplate="%{text} <br>" +
        "Stroke Mortality Rate: %{z}<br>" +
        "<extra></extra>" + "%{text} of population<br>" +
        "<extra></extra>",
        text=map_data['State']
    ))

    fig.update_geos(scope="usa")

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
