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


    # Define a variable for percent with hypertension
    percent_with_hypertension = (
        hyp_map_data['Percent_with_hypertension']).round(2)

    fig.add_trace(go.Scattergeo(
        name="Percent with Hypertension",
        locations=hyp_map_data['STUSPS'],
        locationmode='USA-states',
        lon=[-98.5] * len(hyp_map_data),
        lat=[39.8] * len(hyp_map_data),
        marker=dict(
            size=hyp_map_data['Percent_with_hypertension'],
            sizemode='diameter',
            sizeref=10000000000 *
            hyp_map_data['Percent_with_hypertension'].max() / (7 ** 14),
            color=hyp_map_data['Percent_with_hypertension'],
            colorscale='greys'
        ),
        hoverinfo='none'
    ))

    fig.add_trace(go.Choropleth(
        name="Stroke Mortality Rate",
        locations=map_data['STUSPS'],
        z=map_data['stroke_mortality_rate'],
        locationmode="USA-states",
        colorscale="Viridis_r",
        zmax=map_data['stroke_mortality_rate'].max(),
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar_title="Stroke Mortality Rate",
        # Use the percent_with_hypertension variable in the hovertemplate
        hovertemplate="%{text}<br>" +
        "Hypertension Rate: %{customdata:.2f}%<br>" +
        "Stroke Mortality Rate: %{z:.2f}%<br>" +
        "<extra></extra>",
        # Use the percent_with_hypertension variable as customdata
        customdata=percent_with_hypertension,
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


    fig.update_layout(
        legend=dict(
            orientation="h",
            title={
                'text': 'My Legend Title',
                'font': {
                    'family': 'Times New Roman',
                    'size': 18,
                    'color': 'red'
                }
            },
            font=dict(
                family="Courier",
                size=22,
                color="black"
            ),
            traceorder="reversed"
        )
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
