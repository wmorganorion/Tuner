import pandas as pd
import numpy as np
import includes.graphics.visuals as vs
from IPython.display import display # Allows for the use of display() for DataFrames
from configurations import root_path, prn_out


def load_data():
    
    # Load the Census dataset
    data = pd.read_csv(root_path + "/data/census.csv")

    # Success - Display the first record
    display(data.head(n=1))

    # Total number of records
    n_records = len(data)

    # Number of records where individual's income is more than $50,000
    n_greater_50k = len(data[data["income"] == ">50K"])

    # Number of records where individual's income is at most $50,000
    n_at_most_50k = len(data[data["income"] == "<=50K"])

    # Percentage of individuals whose income is more than $50,000
    greater_percent = (n_greater_50k/n_records) * 100

    if prn_out == True:
        # Print the results
        print("Total number of records: {}".format(n_records))
        print("Individuals making more than $50,000: {}".format(n_greater_50k))
        print("Individuals making at most $50,000: {}".format(n_at_most_50k))
        print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))


    # Split the data into features and target label
    income_raw = data['income']
    features_raw = data.drop('income', axis = 1)

    if prn_out == True:
        # Visualize skewed continuous features of original data
        vs.distribution(data)

    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    if prn_out == True:
        # Visualize the new log distributions
        vs.distribution(features_log_transformed, transformed = True)

    return features_log_transformed, income_raw, n_greater_50k, n_records, features_raw
