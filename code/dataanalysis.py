"""
This file defines test suites for the datacleanup module, which contains
function for data import and transformation.
"""
from typing import Dict
import datacleanup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import operator


def data_set_exploration(df: pd.DataFrame, dir: str):
    """
    This function takes a Pandas dataframe and a directory path as input and prints
    the shape of the dataframe, columns name, and value count of the stroke column.
    It then calls the plot_missing_data and plot_stroke_proportion functions to
    plot and save the graphs in the specified directory.
    """

    print("Data Shape: ", df.shape)
    print("Columns: ")
    for column in df.columns:
        print(column)
    print(df["stroke"].value_counts())
    plot_missing_data(df, dir)
    plot_stroke_proportion(df, dir)


def plot_stroke_proportion(df, dir: str):
    """
    This function takes a Pandas dataframe and a directory path as input and plots
    the proportion of stroke samples in the dataset. It saves the plot in the
    specified directory.
    """

    plt.pie(
        pd.value_counts(df["stroke"]),
        labels=["No Stroke", "Stroke"],
        autopct="%.2f%%",
    )
    plt.title("Proportion Of Stroke Samples")
    plt.savefig(dir + "/strokeProportion.png", bbox_inches="tight")
    plt.clf()


def plot_missing_data(df, dir: str):
    """
    This function takes a Pandas dataframe and a directory path as input and plots
    the number of missing values in each column of the dataset using a heatmap. It
    saves the plot in the specified directory.
    """

    plt.title("Missing Value Status", fontweight="bold")
    ax = sns.heatmap(df.isna().sum().to_frame(), annot=True, fmt="d")
    ax.set_xlabel("Amount of null value")
    plt.savefig(dir + "/nullPoint.png", bbox_inches="tight")
    plt.clf()


def plot_historgram_with_cate(df, dir, cate):
    """
    Method to plot a histogram based on a categorical feature with strokes as the target variable
    """
    sns.kdeplot(
        data=df[df["stroke"] == 0], x=cate, shade=True, alpha=1, label="No Stroke"
    )
    sns.kdeplot(
        data=df[df["stroke"] == 1], x=cate, shade=True, alpha=0.8, label="Stroke"
    )
    plt.legend()
    plt.savefig("" + dir + "/" + cate + "strokeHistorgram.png", bbox="tight")
    plt.clf()


def plot_further_explore_age(df, dir: str):
    """
    Method to plot a countplot based on age categories and strokes as the
    target variable
    """
    df = df.copy()
    df["age_cat"] = pd.cut(
        df["age"],
        bins=[0, 13, 18, 45, 60, 200],
        labels=["Children", "Teens", "Adults", "Mid Adults", "Elderly"],
    )
    sns.countplot(data=df, x="age_cat", hue="stroke")
    plt.savefig(dir + "/ageExplore.png", bbox="tight")
    plt.clf()


def plot_correlation(df, dir: str):
    """
    Method to plot a lower triangle correlation matrix for all features in the
    data
    """
    df = datacleanup.data_transformation_correlation(df)

    # correlation map for all the features
    df_corr = df.drop(columns=["id"]).corr()

    # mask to remove upper triangle
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#f6f5f5")
    ax.set_facecolor("#f6f5f5")

    mask = mask[1:, :-1]
    corr = df_corr.iloc[1:, :-1].copy()

    # plot heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=-0.15,
        vmax=0.5,
        cbar_kws={
            "shrink": 0.5,
        },
        ax=ax,
        linewidth=1,
        linecolor="#f6f5f5",
        square=True,
        annot_kws={"font": "serif", "size": 10, "color": "black"},
    )
    # yticks
    ax.tick_params(axis="y", rotation=0)
    yticks = [
        "Age",
        "Hyper tension",
        "Heart Disease",
        "Marriage",
        "Work",
        "Residence",
        "Glucose Level",
        "BMI",
        "Smoking Status",
        "Stroke",
    ]
    xticks = [
        "Gender",
        "Age",
        "Hyper tension",
        "Heart Disease",
        "Marriage",
        "Work",
        "Residence",
        "Glucose Level",
        "BMI",
        "Smoking Status",
    ]
    ax.set_yticklabels(
        yticks, {"font": "serif", "size": 10, "weight": "bold"}, rotation=0, alpha=0.9
    )
    ax.set_xticklabels(
        xticks, {"font": "serif", "size": 10, "weight": "bold"}, rotation=90, alpha=0.9
    )
    plt.savefig(dir + "/CorrelationMatrix.png", bbox_inches="tight")
    plt.clf()


def normalize_data(df):
    """
    Since there are many observations of no stroke compared to stroke,
    we return normalized data by shuffling the data and reducing the
    number of no stroke data to 1000.
    """
    shuffled_data = df.sample(frac=1, random_state=4)
    stroke_df = shuffled_data.loc[shuffled_data["stroke"] == 1]
    non_stroke_df = df.loc[df["stroke"] == 0].sample(n=1000, random_state=101)
    normalized_stroke = pd.concat([stroke_df, non_stroke_df])
    return normalized_stroke


def cramers_V(var1: str, var2: str) -> float:
    """
    returns the correlation between 2 binary observations
    """
    # Cross table building
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    # Keeping of the test statistic of the Chi2 test
    stat = chi2_contingency(crosstab)[0]
    # Number of observations
    obs = np.sum(crosstab)
    # Take the minimum value between the columns and
    # the rows of the cross table
    mini = min(crosstab.shape) - 1
    return stat / (obs * mini)


def find_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the correlation matrix
    """
    corr_dict = {}
    for col in df.columns:
        corr = (col, cramers_V(df[col], df["stroke"]))
        corr_dict[col] = corr[1]
    return corr_dict


def sorted_correlations(corr_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Sort the correlation coefficients in descending order.
    Return the sorted correlation coefficients.
    """
    sorted_d = dict(sorted(corr_dict.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_d


def find_risk_factor_correlation(risk_factor_df: pd.DataFrame) -> float:
    corr_matrix = risk_factor_df.corr()
    stroke_corr = corr_matrix["stroke"]
    stroke_corr_sorted = stroke_corr.abs().sort_values(ascending=False)
    return stroke_corr_sorted


def comparison_bar_charts(risk_factor_df: pd.DataFrame) -> None:
    """
    This function will plot all categorical variable vs stroke in a bar graph
    after clean up
    """
    # bar chart 1: stroke vs age
    over_65_stroke = sns.countplot(
        x="stroke", hue="over_65", data=risk_factor_df, palette="Set1"
    )
    over_65_stroke.set_xticklabels(["No", "Yes"])
    over_65_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Age", labels=["Under 65", "Over 65"])
    plt.title("Age and Stroke")
    plt.savefig("question1images/over_65.png", bbox_inches="tight")
    plt.clf()

    # bar chart 2: stroke vs smoking
    smoking_status_stroke = sns.countplot(
        x="stroke", hue="smoking_status", data=risk_factor_df, palette="Set1"
    )
    smoking_status_stroke.set_xticklabels(["No", "Yes"])
    smoking_status_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Smoking Status", labels=["Not a Smoker", "Smoker", "Unknown"])
    plt.title("Smoking Status and Stroke")
    plt.savefig("question1images/smoking_status.png", bbox_inches="tight")
    plt.show()
    plt.clf()

    # bar chart 3: stroke vs gender
    gender_stroke = sns.countplot(
        x="stroke", hue="gender", data=risk_factor_df, palette="Set1"
    )
    gender_stroke.set_xticklabels(["No", "Yes"])
    gender_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Gender", labels=["Female", "Male"])
    plt.title("Gender and Stroke")
    plt.savefig("question1images/gender.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 4: stroke vs heart disease
    heart_disease_stroke = sns.countplot(
        x="stroke", hue="heart_disease", data=risk_factor_df, palette="Set1"
    )
    heart_disease_stroke.set_xticklabels(["No", "Yes"])
    heart_disease_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(
        title="Status", labels=["did not have heart disease", "had heart disease"]
    )
    plt.title("Heart Disease and Stroke")
    plt.savefig("question1images/heart_disease.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 5: stroke vs hypertension
    hypertension_and_stroke = sns.countplot(
        x="stroke", hue="hypertension", data=risk_factor_df, palette="Set1"
    )
    hypertension_and_stroke.set_xticklabels(["No", "Yes"])
    hypertension_and_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Status", labels=["did not have hypertension", "had hypertension"])
    plt.title("Hypertension and Stroke")
    plt.savefig("question1images/hypertension.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 6: stroke vs married
    marital_status_stroke = sns.countplot(
        x="stroke", hue="married", data=risk_factor_df, palette="Set1"
    )
    marital_status_stroke.set_xticklabels(["No", "Yes"])
    marital_status_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Marital Status", labels=["not married", "married"])
    plt.title("Marital Status and Stroke")
    plt.savefig("question1images/Marital_status.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 7: stroke vs high glucose
    high_glucose_stroke = sns.countplot(
        x="stroke", hue="high_glucose", data=risk_factor_df, palette="Set1"
    )
    high_glucose_stroke.set_xticklabels(["No", "Yes"])
    high_glucose_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Status", labels=["normal glucose", "high glucose"])
    plt.title("Glucose level and Stroke")
    plt.savefig("question1images/glucose.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 8: stroke vs high BMI
    high_bmi_stroke = sns.countplot(
        x="stroke", hue="high_BMI", data=risk_factor_df, palette="Set1"
    )
    high_bmi_stroke.set_xticklabels(["No", "Yes"])
    high_bmi_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Status", labels=["normal BMI", "high BMI"])
    plt.title("High BMI and Stroke")
    plt.savefig("question1images/high_bmi.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 9: stroke vs low BMI
    low_bmi_stroke = sns.countplot(
        x="stroke", hue="low_BMI", data=risk_factor_df, palette="Set1"
    )
    low_bmi_stroke.set_xticklabels(["No", "Yes"])
    low_bmi_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="Status", labels=["normal BMI", "low BMI"])
    plt.title("low BMI and Stroke")
    plt.savefig("question1images/low_bmi.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart 10: stroke vs residence
    residence_stroke = sns.countplot(
        x="stroke", hue="residence", data=risk_factor_df, palette="Set1"
    )
    residence_stroke.set_xticklabels(["No", "Yes"])
    residence_stroke.set(xlabel="Had Stroke", ylabel="Count")
    plt.legend(title="residence type", labels=["rural", "urban"])
    plt.title("Residence Type and Stroke")
    plt.savefig("question1images/residence_type.png", bbox_inches="tight")
    # plt.show()
    plt.clf()

    # bar chart plotting when stroke = 1


def main():
    risk_factor_data = datacleanup.create_risk_factor_df("datasets/stroke_data_1.csv")
    print("Data Summary")
    # data summary
    df = pd.read_csv("datasets/stroke_data_1.csv")
    print(df.columns)
    data_set_exploration(df, dir="dataSummary")

    # Distribution for numerical value
    for column in ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]:
        plot_historgram_with_cate(df, "dataSummary", column)

    # Question 1
    print(sorted_correlations(find_correlations(df)))
    plot_further_explore_age(df, dir="question1Images")

    # Explore more on the 1st factor that have highest correlation with stroke
    plot_historgram_with_cate(df, "question1Images", "age")
    plot_correlation(df, dir="question1Images")

    # visualization_correlation_matrix(df)
    # pair_visualization(df)
    normalized_risk_factor = normalize_data(risk_factor_data)
    correlations = find_correlations(normalized_risk_factor)
    print(sorted_correlations(correlations))


if __name__ == "__main__":
    main()
