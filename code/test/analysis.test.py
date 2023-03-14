"""
This file defines test suites for the dataanalysis module, which contains
function for data analysis, exploration, and visualization.
"""
import sys
import os
import dataanalysis
from utilsTest import create_sample_testing_df


# getting current directory
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
# import the module with relative path

# Output directory for testing graph
TEST_DIR = "code/test/testGraphOutput"


def test_data_set_exploration():
    """
    This function tests the data_set_exploration function in the dataanalysis
    module. It creates a sample testing dataframe and applies the function to
    explore the dataset. The output is then checked by hand.
    """
    df = create_sample_testing_df()
    dataanalysis.data_set_exploration(df, dir=TEST_DIR)
    # Check the output by hand


def test_plot_missing_data():
    """
    This function tests the plot_missing_data function in the dataanalysis
    module. It creates a sample testing dataframe and applies the function to
    visualize missing data. The output is then checked by hand.
    """
    df = create_sample_testing_df()
    dataanalysis.plot_missing_data(df, dir=TEST_DIR)
    # Check the output by hand


def test_plot_stroke_proportion():
    """
    This function tests the plot_stroke_proportion function in the
    dataanalysis module. It creates a sample testing dataframe and applies
    the function to visualize the proportion of stroke in the dataset.
    The output is then checked by hand.
    """
    df = create_sample_testing_df()
    dataanalysis.plot_stroke_proportion(df, dir=TEST_DIR)
    # Check the output by hand


def test_plot_historgram_with_cate():
    """
    This function tests the plot_historgram_with_cate function in the
    dataanalysis module. It creates a sample testing dataframe and applies
    the function to visualize a histogram of ages with stroke as a categorical
    variable. The output is saved in the specified directory and is checked
    to confirm that the plot was saved.
    """
    df = create_sample_testing_df()
    dataanalysis.plot_historgram_with_cate(df, TEST_DIR, "age")
    # Check that the plot was saved in the specified directory


def test_plot_further_explore_age():
    """
    This function tests the plot_further_explore_age function in the
    dataanalysis module. It creates a sample testing dataframe and applies
    the function to further explore age distribution with respect to stroke.
    The output is saved in the specified directory and is checked to confirm
      that the plot was saved.
    """
    df = create_sample_testing_df()
    dataanalysis.plot_further_explore_age(df, dir=TEST_DIR)
    # Check that the plot was saved in the specified directory


def test_plot_correlation():
    """
    This function tests the plot_correlation function in the dataanalysis
    module. It creates a sample testing dataframe and applies the function
    to visualize correlation between variables in the dataset. The output
    is saved in the specified directory and is checked to confirm that
    the plot was saved.
    """
    df = create_sample_testing_df()
    dataanalysis.plot_correlation(df, dir=TEST_DIR)
    # Check that the plot was saved in the specified directory


def main():
    """
    Main function to call all test
    """
    test_data_set_exploration()
    test_plot_missing_data()
    test_plot_stroke_proportion()
    test_plot_historgram_with_cate()
    test_plot_further_explore_age()
    test_plot_correlation()


if __name__ == "__main__":
    main()
