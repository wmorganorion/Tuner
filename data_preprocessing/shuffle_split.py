# Import train_test_split
from sklearn.cross_validation import train_test_split
from configurations import prn_out



def data_aug_trn_vld_tst(features_final, income):
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                        income, 
                                                        test_size = 0.2, 
                                                        random_state = 0)

    # Show the results of the split
    if prn_out == True:
        print("Training set has {} samples.".format(X_train.shape[0]))
        print("Testing set has {} samples.".format(X_test.shape[0]))

    return X_train, X_test, y_train, y_test
