# Import 'GridSearchCV', 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from configurations import prn_out, feat_imp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from includes.feature_importance import feat_importance

def grid_search(X_train,y_train, X_test, y_test):
    # Initialize the classifier
    clf_A = DecisionTreeClassifier(random_state=0)
    clf_B = SVC(random_state=0)
    clf_C = RandomForestClassifier(random_state=0)    
    
    # Create the parameters list you wish to tune, using a dictionary if needed.
    parameters_A = {'criterion': ['gini']}
    
    parameters_B = {'kernel':['rbf','linear','sigmoid']}
    
    parameters_C = {'n_estimators': [10,100,500,1000,2000], 
                    'criterion': ['gini', 'entropy']}
      
    w = [clf_A, clf_B, clf_C]
    
    z = (parameters_A, parameters_B, parameters_C)
  
    i=0
    while i != len(w):
        if prn_out == True:
                print("")
                print("Grid Search in progress ...")
                print("")
                
        # Make an fbeta_score scoring object using make_scorer()
        scorer = make_scorer(fbeta_score, beta=0.5)
    
        clf = w[i]
        parameters = z[i]        
    
        # Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
        grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
        i +=1
        
        # Fit the grid search object to the training data and find the optimal parameters using fit()
        grid_fit = grid_obj.fit(X_train,y_train)
    
        # Get the estimator
        best_clf = grid_fit.best_estimator_
    
        # Make predictions using the unoptimized and model
        predictions = (clf.fit(X_train, y_train)).predict(X_test)
        best_predictions = best_clf.predict(X_test)
    
        # Report the before-and-afterscores
        if prn_out == True:
            print("Model Name: " + clf.__class__.__name__)
            print("Unoptimized model\n------")
            print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
            print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
            print("\nOptimized Model\n------")
            print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
            print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
            print("")
            
        if feat_imp == True:
             feat_importance(best_clf, X_train, y_train, X_test, y_test, best_predictions)        


    return best_clf, best_predictions
