import os


global root_path
global prn_out
global load_data
global preprocess_data
global shuffle_split
global naive_pred
global tp_pipeln
global gd_sch
global feat_imp


# Turn printing & display of images to screen, ON/OFF.
prn_out = True

# Determine root path for application.
root_path = os.getcwd()

# Turn each section of Tuner ON/OFF based on need.
load_data = True
preprocess_data = True
shuffle_split = True
naive_pred = True

# Only option 1 is completed. Next version will include 2 & 3 options
# Select Supervised [1], Unsupervised [2] or Transfer Learning [3]
tp_pipeln = 1

# Turn grid search ON/OFF.
gd_sch = True

# Feature Importance ON/OFF.
feat_imp = True
