from configurations import prn_out
#from model_selection.functions import *
import includes.graphics.visuals as vs

# Import the supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from model_selection.predict_pl import train_predict



def sup_predict(X_train, y_train, X_test, y_test):

    # Initialize the three models
    clf_A = DecisionTreeClassifier(random_state=0)
    clf_B = SVC(random_state=0)
    clf_C = RandomForestClassifier(random_state=0)

    # Calculate the number of samples for 1%, 10%, and 100% of the training data
    samples_100 = int(len(X_train))
    samples_10 = int(.1*len(X_train))
    samples_1 = int(.01*len(X_train))

    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    if prn_out == True:
        # Run metrics visualization for the three supervised learning models chosen
        vs.evaluate(results, accuracy, fscore)
        


