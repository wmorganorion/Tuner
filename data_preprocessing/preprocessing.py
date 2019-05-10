import pandas as pd
import numpy as np

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler
from configurations import prn_out


def pre_process_data(features_log_transformed, income_raw):
    
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # Show an example of a record with scaling applied
    display(features_log_minmax_transform.head(n = 5))


    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_log_minmax_transform)

    # Encode the 'income_raw' data to numerical values
    income = income_raw.map(lambda x:0 if x== "<=50K" else 1)

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)

    if prn_out == True:
        print("{} total features after one-hot encoding.".format(len(encoded)))

    # Uncomment the following line to see the encoded feature names
    if prn_out == True:
        print (encoded)

    return features_final, income
