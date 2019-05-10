from configurations import prn_out

'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''

def naive_pred(n_greater_50k, n_records):

    # Calculate accuracy, precision and recall
    accuracy = 1.0 * n_greater_50k / n_records
    recall = 1.0
    precision = accuracy
    beta=0.5

    # Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    fscore = (1 + beta*beta) * ((1.0*precision*recall)/((beta*beta*precision) + recall))

    if prn_out == True:
        # Print the results
        print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

    return accuracy, fscore
