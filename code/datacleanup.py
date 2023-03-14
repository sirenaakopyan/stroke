"""
This file contains functions that clean up the csv dataset
and attempt to find the top 3 demographic and health risk
factors for stroke.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from functools import reduce


from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_risk_factor_df(stroke_data: str) -> pd.DataFrame:
    """
    Cleans the stroke data file. Converts all columns of interest to binary.
    To convert data in gender column. Male = 1 and Female = 0
    we do not worry about other genders because they are not
    included in the dataset. To convert data in smoking_status column:
    has smoked = 1.
    Never smoked = 0. To convert BMI, according to the CDC
    If your BMI is less than 18.5, it falls within the underweight range.
    If your BMI is 18.5 to <25, it falls within the healthy weight range.
    If your BMI is 25.0 to <30, it falls within the overweight range.
    If your BMI is 30.0 or higher, it falls within the obesity range.
    To convert avg_glucose_level into binary 99 or below is 0, 100+ is 1.
    To convert age into binary less than 65 is 0 , 65+ is 1
    To convert residence_type to binary 1 for urban 0 for rural.
    To convert ever_married to binary 1 for married 0 for not married
    """
    unclean_data = pd.read_csv(stroke_data)
    cleaned_gender = unclean_data.copy()
    cleaned_gender.loc[cleaned_gender["gender"] == "Male", "gender"] = 1
    cleaned_gender.loc[cleaned_gender["gender"] == "Female", "gender"] = 0
    cleaned_smoking = cleaned_gender.copy()
    cleaned_smoking.loc[
        cleaned_smoking["smoking_status"] == "formerly smoked", "smoking_status"
    ] = 1
    cleaned_smoking.loc[
        cleaned_smoking["smoking_status"] == "smokes", "smoking_status"
    ] = 1
    cleaned_smoking.loc[
        cleaned_smoking["smoking_status"] == "never smoked", "smoking_status"
    ] = 0
    cleaned_BMI = cleaned_smoking.copy()
    cleaned_BMI["low_BMI"] = np.where(cleaned_BMI["bmi"] < 18.5, 1, 0)
    cleaned_BMI["high_BMI"] = np.where(
        (cleaned_BMI["bmi"] >= 18.5) & (cleaned_BMI["bmi"] <= 24.9),
        0,
        np.where((cleaned_BMI["bmi"] >= 25) & (cleaned_BMI["bmi"] <= 29.9), 1, 0),
    )
    cleaned_glucose = cleaned_BMI.copy()
    cleaned_glucose["high_glucose"] = np.where(
        (cleaned_glucose["avg_glucose_level"] <= 99.9),
        0,
        np.where(
            (cleaned_glucose["avg_glucose_level"] > 100.001),
            1,
            cleaned_glucose["avg_glucose_level"],
        ),
    ).astype(int)
    cleaned_age = cleaned_glucose.copy()
    cleaned_age["over_65"] = np.where(
        (cleaned_age["age"] <= 64),
        0,
        np.where((cleaned_age["age"] >= 65), 1, cleaned_age["age"]),
    ).astype(int)
    cleaned_residence = cleaned_age.copy()
    cleaned_residence.loc[
        cleaned_residence["Residence_type"] == "Urban", "residence"
    ] = 1
    cleaned_residence.loc[
        cleaned_residence["Residence_type"] == "Rural", "residence"
    ] = 0
    cleaned_married = cleaned_residence.copy()
    cleaned_married.loc[cleaned_married["ever_married"] == "Yes", "married"] = 1
    cleaned_married.loc[cleaned_married["ever_married"] == "No", "married"] = 0
    final_df = cleaned_married[
        [
            "hypertension",
            "heart_disease",
            "high_glucose",
            "low_BMI",
            "high_BMI",
            "gender",
            "over_65",
            "married",
            "residence",
            "smoking_status",
            "stroke",
        ]
    ]
    return final_df


def data_transformation_correlation(df):
    # feature log transformations (since data is skewed -> transform back to
    # normal distribution, would not affect the correlation)
    # df.fillna(29, inplace = True)
    df["age"] = df["age"].apply(lambda x: np.log(x + 10) * 3)
    df["avg_glucose_level"] = df["avg_glucose_level"].apply(
        lambda x: np.log(x + 10) * 2
    )
    df["bmi"] = df["bmi"].apply(lambda x: np.log(x + 10) * 2)

    # preprocessing - label enconding and numerical value scaling
    ss = StandardScaler()
    le = LabelEncoder()

    # label encoding of ordinal categorical features
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    cols = df.columns
    # normalizing with standard scaler of numerical features
    df[cols] = ss.fit_transform(df[cols])
    return df


def create_shapefile_for_bubble_map(
    shapefile: str,
    hypertension: str,
    obesity: str,
    diabetes: str,
    abbr: str,
    stroke: str,
) -> gpd.GeoDataFrame:
    """
    Plotly bubble map to layer map of strokes over map of each health risk
    factor
    """
    # create shapefile for geometry and state name
    us_shapefile = gpd.read_file(shapefile)
    us_shapefile = us_shapefile.rename(columns={"NAME": "State"})

    # hypertension by state
    hypertension_state = pd.read_excel(hypertension, engine="openpyxl")
    hypertension_state = hypertension_state[
        hypertension_state["State"] != "Virgin Islands"
    ]

    # Obesity (high-BMI by state)
    obesity_state = pd.read_csv(obesity)
    obesity_state = obesity_state[obesity_state["State"] != "Virgin Islands"]
    obesity_state = obesity_state[["State", "Prevalence"]]
    obesity_state = obesity_state.rename(columns={"Prevalence": "Obesity_prev_perc"})

    # high-glucose (Diabetes by state)
    raw_diabetes_state = pd.read_csv(diabetes)
    state_abbr = pd.read_csv(abbr)
    state_abbr = state_abbr[["code", "state"]]
    raw_diabetes_state = raw_diabetes_state.groupby("state_abbr", as_index=False).mean()
    diabetes_state = state_abbr.merge(
        raw_diabetes_state, left_on="code", right_on="state_abbr", how="outer"
    )
    diabetes_state = diabetes_state[diabetes_state["state"] != "Virgin Islands"]
    diabetes_state = diabetes_state[["state", "value"]]
    diabetes_state = diabetes_state.rename(
        columns={"value": "Diabetes_prev_perc", "state": "State"}
    )

    # stroke mortality by state
    stroke_mortality_df = pd.read_csv(stroke)
    year_2020 = stroke_mortality_df["YEAR"] == 2020
    stroke_mortality_df = stroke_mortality_df[year_2020]
    stroke_mortality_df = stroke_mortality_df[
        stroke_mortality_df["STATE"] != "Virgin Islands"
    ]
    stroke_mortality_df = stroke_mortality_df[["STATE", "RATE"]]
    stroke_mortality_df = state_abbr.merge(
        stroke_mortality_df, left_on="code", right_on="STATE", how="outer"
    )
    stroke_mortality_df = stroke_mortality_df[
        stroke_mortality_df["state"] != "Virgin Islands"
    ]
    stroke_mortality_df = stroke_mortality_df[["state", "RATE"]]
    stroke_mortality_df = stroke_mortality_df.rename(
        columns={"RATE": "stroke_mortality_rate", "state": "State"}
    )

    # combine datasets
    dataframes_to_merge = [
        hypertension_state,
        obesity_state,
        diabetes_state,
        stroke_mortality_df,
    ]
    merged_risk_factors = reduce(
        lambda left, right: pd.merge(left, right, on=["State"], how="outer"),
        dataframes_to_merge,
    )

    excluded_states = [
        "United States Virgin Islands",
        "Commonwealth of the Northern Mariana Islands",
        "Guam",
        "American Samoa",
        "District of Columbia",
        "Puerto Rico",
    ]  # list of state codes to exclude
    us_shapefile = us_shapefile[~us_shapefile["State"].isin(excluded_states)]
    merged_risk_factors = merged_risk_factors[
        ~merged_risk_factors["State"].isin(excluded_states)
    ]

    risk_factors_and_stroke_df = us_shapefile.merge(
        merged_risk_factors, left_on="State", right_on="State", how="inner"
    )

    return risk_factors_and_stroke_df
