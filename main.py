# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:10:35 2019

@author: wmorgan
"""

# Import libraries
import configurations as cf
import includes.naive_predictor as inc_np
import data_preprocessing as dppd
import includes.graphics.visuals as vs

from time import ctime
#from model_selection.supervised import sup_predict

# Import the models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from kmodes import kmodes

from model_selection.predict_pl import train_predict
from includes.gridsearch import grid_search
from includes.feature_importance import feat_importance



if __name__ == '__main__':
    
    if cf.prn_out == True:
            start_time = ctime()
            print("")
            print("Program Name: Tuner | Program Start Time :", start_time)
            print("")
   
# load data
    if cf.load_data == True:        
        try:
            features_log_transformed, income_raw, n_greater_50k, n_records, features_raw = dppd.process_data.load_data()
            
        except Exception as e:
            print(str(e))
                  
        
# preprocess
    if cf.preprocess_data == True:        
        try:
            features_final, income = dppd.preprocessing.pre_process_data(features_log_transformed, income_raw)
            
        except Exception as e:
            print(str(e))
            
        
# determine split for training, validation, testing data sets
    if cf.shuffle_split == True:        
        try:
            X_train, X_test, y_train, y_test = dppd.shuffle_split.data_aug_trn_vld_tst(features_final, income)
            
        except Exception as e:
            print(str(e))
         
# naive predictor
    if cf.naive_pred == True:        
        try:   
            accuracy, fscore = inc_np.naive_predictor.naive_pred(n_greater_50k, n_records)
            
        except Exception as e:
            print(str(e))
            
# Training and Predicting Pipeline
    try:
        if cf.tp_pipeln == 1:
            # Initialize the Supervised models
            clf_A = DecisionTreeClassifier(random_state=0)
            clf_B = SVC(random_state=0)
            clf_C = RandomForestClassifier(random_state=0)
            
        if cf.tp_pipeln == 2:
            # Initialize the Un-supervised models
            clf_A = GMM()
            clf_B = KMeans(n_clusters=2, random_state=0)
            clf_C = kmodes()
            
#        if cf.tp_pipeln == 3:
#            # Initialize the Transfer Learning models
#            clf_A = 
#            clf_B = 
#            clf_C = 
                    
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

        if cf.prn_out == True:
            # Run metrics visualization for the three models chosen
            vs.evaluate(results, accuracy, fscore)
      
    except Exception as e:
        print(str(e))
        

    if cf.gd_sch == True:
        try:
#            if cf.prn_out == True:
#                print("Grid Search in progress ...")
#                print("")
            
            best_clf, best_predictions = grid_search(X_train, y_train, X_test, y_test)
            
        except Exception as e:
            print(str(e))
        
#    if cf.feat_imp == True:
#        try:
#            if cf.prn_out == True:
#                print("")
#                print("Feature Importance in progress ...")
#                print("")
#            
#            feat_importance(best_clf, X_train, y_train, X_test, y_test, best_predictions)
#        
#        except Exception as e:
#            print(str(e))
        
                   
    if cf.prn_out == True:
            end_time = ctime()
            print("")
            print("Program End Time :", end_time)       

    
