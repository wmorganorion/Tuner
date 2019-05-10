import numpy as np
import includes.graphics.visuals as vs

# Import functionality for cloning a model
from sklearn.base import clone
from configurations import prn_out
from time import time
from sklearn.metrics import accuracy_score, fbeta_score


def feat_importance(best_clf, X_train, y_train, X_test, y_test, best_predictions):

    if prn_out == True:
        print("")
        print("Feature Importance in progress ...")
        print("")
                
    start = time()
    best_clf.predict(X_test)
    end = time()

    if prn_out == True:
        print("clf took " + str(end-start))

    start = time()
    best_clf.predict(X_test)
    end = time()

    if prn_out == True:
        print("best_clf took " + str(end-start))

    # capture 
    model = best_clf

    # Extract the feature importances using .feature_importances_ 
    try:
        importances = model.feature_importances_
    except:
        print("No features. Look into Coef.")
        print("")
        return
    
    # Plot
    if prn_out == True:
        vs.feature_plot(importances, X_train, y_train)

    # Reduce the feature space
    X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
    X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

    # Train on the "best" model found from grid search earlier
    clf = (clone(best_clf)).fit(X_train_reduced, y_train)

    # Make new predictions
    reduced_predictions = clf.predict(X_test_reduced)

    # Report scores from the final model using both versions of data
    if prn_out == True:
        print("Final Model trained on full data\n------")
        print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
        print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
        print("\nFinal Model trained on reduced data\n------")
        print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
        print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
