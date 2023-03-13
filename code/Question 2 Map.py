"""
Question 2.
Research Questions: 

1. What is the prevalence of diabetes, hypertension, and obesity in different 
states across the US?

2. Is there a correlation between the prevalence of these risk factors and 
stroke mortality rates in each state?


This code defines three functions that display bubble maps of hypertension, 
obesity, and diabetes rates across the United States over a choropleth map
colored by stroke mortality rates.
"""

import datacleanup
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.subplots as sp


def map_hypertension(map_data: pd.DataFrame):
    """
    Displays a bubble map of the hypertension rates across the 
    country over a choropleth map that is colored by stroke mortality
    rates.

    Parameters:
    map_data (pd.DataFrame): A pandas DataFrame containing the data 
    to be plotted. It should include columns for state abbreviation, 
    stroke mortality rate, and percent with hypertension.

    Returns:
    None
    """

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


def map_obesity(map_data: pd.DataFrame):
    """
    Displays a bubble map of the obesity rates across the 
    country over a choropleth map that is colored by stroke mortality
    rates.

    Parameters:
    map_data (pd.DataFrame): A pandas DataFrame containing the data 
    to be plotted. It should include columns for state abbreviation, 
    stroke mortality rate, and percent with obesity.

    Returns:
    None
    """

    print(map_data)
    print(map_data.columns)

    fig = go.Figure()

    map_data = map_data.replace("No data", np.nan)
    ob_map_data = map_data.copy()
    ob_map_data['Obesity_prev_perc'] = pd.to_numeric(
        ob_map_data['Obesity_prev_perc'], errors='coerce')
    ob_map_data = ob_map_data.dropna(subset=['Obesity_prev_perc'])


    # Define a variable for percent with obesity
    percent_with_obesity = (
        ob_map_data['Obesity_prev_perc']).round(2)

    fig.add_trace(go.Scattergeo(
        name="Percent with Obesity",
        locations=ob_map_data['STUSPS'],
        locationmode='USA-states',
        lon=[-98.5] * len(ob_map_data),
        lat=[39.8] * len(ob_map_data),
        marker=dict(
            size=ob_map_data['Obesity_prev_perc'],
            sizemode='diameter',
            sizeref=10000000000 *
            ob_map_data['Obesity_prev_perc'].max() / (7 ** 14),
            color=ob_map_data['Obesity_prev_perc'],
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
        # Use the percent_with_obesity variable in the hovertemplate
        hovertemplate="%{text}<br>" +
        "Obesity Rate: %{customdata:.2f}%<br>" +
        "Stroke Mortality Rate: %{z:.2f}%<br>" +
        "<extra></extra>",
        # Use the percent_with_obesity variable as customdata
        customdata=percent_with_obesity,
        text=map_data['State']
    ))

    fig.update_geos(scope="usa")

    fig.update_layout(
        title={
            "text": "Stroke Mortality and Obesity by State",
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


def map_diabetes(map_data: pd.DataFrame):
    """
    Displays a bubble map of the diabetes rates across the 
    country over a choropleth map that is colored by stroke mortality
    rates.

    Parameters:
    map_data (pd.DataFrame): A pandas DataFrame containing the data 
    to be plotted. It should include columns for state abbreviation, 
    stroke mortality rate, and percent with diabetes.

    Returns:
    None
    """

    print(map_data)
    print(map_data.columns)

    fig = go.Figure()

    map_data = map_data.replace("No data", np.nan)
    di_map_data = map_data.copy()
    di_map_data['Diabetes_prev_perc'] = pd.to_numeric(
        di_map_data['Diabetes_prev_perc'], errors='coerce')
    di_map_data = di_map_data.dropna(subset=['Diabetes_prev_perc'])

    # Define a variable for percent with diabetes
    percent_with_diabetes = (
        di_map_data['Diabetes_prev_perc'] * 100).round(2)

    fig.add_trace(go.Scattergeo(
        name="Percent with Diabetes",
        locations=di_map_data['STUSPS'],
        locationmode='USA-states',
        lon=[-98.5] * len(di_map_data),
        lat=[39.8] * len(di_map_data),
        marker=dict(
            size=di_map_data['Diabetes_prev_perc'],
            sizemode='diameter',
            sizeref=10000000000 *
            di_map_data['Diabetes_prev_perc'].max() / (7 ** 14),
            color=di_map_data['Diabetes_prev_perc'],
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
        # Use the percent_with_diabetes variable in the hovertemplate
        hovertemplate="%{text}<br>" +
        "Diabetes Rate: %{customdata:.2f}%<br>" +
        "Stroke Mortality Rate: %{z:.2f}%<br>" +
        "<extra></extra>",
        # Use the percent_with_diabetes variable as customdata
        customdata=percent_with_diabetes,
        text=map_data['State']
    ))

    fig.update_geos(scope="usa")

    fig.update_layout(
        title={
            "text": "Stroke Mortality and Diabetes by State",
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
    map_hypertension(map_data)
    map_obesity(map_data)
    map_diabetes(map_data)


if __name__ == '__main__':
    main()