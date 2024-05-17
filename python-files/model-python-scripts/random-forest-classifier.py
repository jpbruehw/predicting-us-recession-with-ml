# import libraries
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc, recall_score

# initialize shap so we can
# visualize the plots
shap.initjs()

# import the training data
data_raw = pd.read_excel('[SET PATH TO TRAINING/TESTING DATA]', index_col=0)

# set up function to loop over and test different
# thresholds then extract the best one
def find_best_threshold(y_predicted_raw, y_true):
    # create range from 0-1
    # in increments of 0.001
    threshold_intervals = np.arange(0, 1.0001, 0.0001)
    # now loop over each interval and run the model
    # to see which one performs best
    # create list to store the results
    results = []
    for threshold in threshold_intervals:
        # compare at each threshold
        y_pred_at_threshold = (y_predicted_raw > threshold).astype(int)
        # calculate the f1 score for the predicted
        threshold_f1 = f1_score(y_true, y_pred_at_threshold)
        # create tuple of each threshold and f1
        results.append((threshold_f1, y_pred_at_threshold, threshold))
    # find the threshold with the max f1
    max_threshold_tuple = max(results, key=lambda x: x[0])
    # now extract the threshold from tuple
    best_threshold = max_threshold_tuple[2]
    y_pred = max_threshold_tuple[1]
    best_f1 = max_threshold_tuple[0]
    # return the result
    return [best_threshold, y_pred, best_f1]

# path to store the exported figures
# roc_curve plots
roc_fig_dir_path = '[SET PATH TO DIRECTORY FOR ROC CURVES]'
# shap plots
shap_fig_dir_path = '[SET PATH TO DIRECTORY FOR SHAP PLOTS]'

# helper function to create roc_curve
def roc_curve_plot(y_true, y_pred_prob, lag, fig_dir_path, model_name):
    # get the fpr, trp, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    # call method on fpr and tpr
    roc_auc = auc(fpr, tpr)
    
    # create plot
    plt.figure(figsize=(10, 6))
    # we insert the area under curve up to 2 decimal places
    # we can insert using the modulo operator - %
    plt.plot(fpr, tpr, color='lightblue', linewidth=2, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkred', linewidth=2, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Logit ROC Curve for {lag} Month Lag")
    plt.legend(loc="lower right")
    file_name = f"roc_curve_{model_name}_lag_{lag}.png"
    save_path = os.path.join(fig_dir_path, file_name)
    plt.savefig(save_path, dpi=300)
    plt.show()
    
# helper function to perform shap analysis
# used these tutorials
# https://shorturl.at/nwGP8
# don't need to pass in training data for tree models
def shap_analysis(model, X_test, fig_dir_path, model_name, lag):
    # get the shap explainer variables
    explainer = shap.TreeExplainer(model)
    # get the raw shap values
    shap_values = explainer.shap_values(X_test)
    # we need to extract the correct values
    # this is a quirk of the rf models' outputs
    # common issue with solution here:
    # https://shorturl.at/fEQ16
    # transform shap values from
    # multidimensional array to a format we can pass in
    transformed_shap_vals = shap_values[:,:,1]
    # create explainer object
    shap_expl = shap.Explanation(transformed_shap_vals, base_values=explainer.expected_value, data=X_test)
    # create file name
    file_name = f"beeswarm_plt_{model_name}_lag_{lag}.png"
    # initialize save path
    save_path = os.path.join(fig_dir_path, file_name)
    # create figure object and set size
    plt.figure(figsize=(10, 6))
    # create beeswarm plot
    shap.plots.beeswarm(shap_expl, show=False)
    plt.tight_layout()
    # save the plot
    plt.savefig(save_path, dpi=300)
    # create file name for summary plot
    summary_file_name = f"summary_plot_{model_name}_lag_{lag}.png"
    summary_save_path = os.path.join(fig_dir_path, summary_file_name)
    # create figure for the summary plot as well
    plt.figure(figsize=(10, 6))
    # create summary plot
    shap.summary_plot(transformed_shap_vals, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    # export plot
    plt.savefig(summary_save_path, dpi=300)

# set up a params grid to find the best performing model
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# set up function to run the the random forest
def run_random_forest(data, lag, test_size, scoring, params):
    
    """
    This function takes various inputs 
    and returns summary statistics
    for the random forest models
    some inputs persist across iterations
    """
    
    # initiate the model
    rf_classifier = RandomForestClassifier(random_state=42, verbose=1, n_jobs=12)
    
    # make a copy of the original DataFrame to avoid modifying it
    data_copy = data.copy()
    
    # modify dataset for lag
    # we want to set the recession indicator back by the lag so that t0 is aligned with t+lag
    data_copy[f"nber_recession_{lag}_month_lag"] = data_copy['nber_recession'].shift(-lag)
    
    # drop the original recession column and na values
    data_copy = data_copy.drop(columns=['nber_recession'])
    data_copy = data_copy.dropna()
    
    # set up training and testing data
    X = data_copy.drop(columns=[f"nber_recession_{lag}_month_lag"])
    y = data_copy[f"nber_recession_{lag}_month_lag"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # set up the grid search object to perform the analysis
    # set cross validation to 5 which is a standard benchmark
    grid_search_cv = GridSearchCV(estimator=rf_classifier, param_grid=params, cv=5, scoring=scoring)
    
    # perform the initial grid search
    grid_search_cv.fit(X_train_resampled, y_train_resampled)
    
    # get the best performing model
    best_parameters = grid_search_cv.best_params_
    best_model = grid_search_cv.best_estimator_
    
    # predict the results
    # this returns two columns, one for the 0 classification
    # and one for the 1 class, so we take all the rows
    # from the second column 
    y_pred_raw = best_model.predict_proba(X_test)[:,1]
    
    # extract the data for the best threshold
    best_model_results = find_best_threshold(y_pred_raw, y_test)
    # extract the relevant results
    threshold = best_model_results[0]
    y_pred = best_model_results[1]
    best_f1 = best_model_results[2]
    
    # create a confusion matrix to visualize results
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # make the roc_curve plot
    # this does not take into account
    # the threshold we ind. calculate
    roc_curve_plot(y_test, y_pred_raw, lag, roc_fig_dir_path, 'rf')
    
    # perform shap analysis on the model
    shap_analysis(best_model, X_test, shap_fig_dir_path, 'rf', lag)

    # get predicted values and metrics
    metrics_obj = {
       'accuracy': accuracy_score(y_test, y_pred),
       'precision': precision_score(y_test, y_pred),
       'recall': recall_score(y_test, y_pred),
       'f1': best_f1,
       'roc_auc': roc_auc_score(y_test, y_pred_raw),
       }

    return {'data': data_copy,
            'best_parameters': best_parameters,
            'best_model': best_model,
            'best_threshold': threshold,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_raw': y_pred_raw,
            'confusion_matrix': conf_mat,
            'model_metrics': metrics_obj}

# set up lag list
lags = [3, 6, 9, 12, 18]

# make a list of the resulting ranom forest
random_forest_results = [(f"{lag}_month_lag_results", run_random_forest(data_raw, lag, 0.2, 'f1', param_grid, False)) for lag in lags]

# make a dataframe of all accuracy results
headers_metrics = ['lag', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'threshold', 'conf_matrix']
# store the results for each iteration
iteration_metrics = []
# iterate over results
for result in random_forest_results:
    # extract from the tuple
    metrics = result[1]['model_metrics']
    # extract each value
    values = [val for _, val in metrics.items()]
    # insert name of lag
    values.insert(0, result[0])
    # get the best threshold value for each lag
    threshold = result[1]['best_threshold']
    # append to values
    values.append(threshold)
    # get the confusion matrix
    conf_matrix = result[1]['confusion_matrix']
    # append to values
    values.append(conf_matrix)
    # append to the list
    iteration_metrics.append(values)
# convert to a dataframe
metric_data = pd.DataFrame(iteration_metrics, columns=headers_metrics)

print(metric_data)


# write data to excel to transfer to local file
# we will do further data processing in another script
path = '~/desktop/master-thesis-code/model-summaries/summary-random-forest-models.xlsx'
writer = pd.ExcelWriter(path, engine='openpyxl')

metric_data.to_excel(writer, sheet_name='random-forest-summary-stats', index=False)

# go through and see if the model is over or underestimating recessions
headers_false_true_summary = ['lag', 'recession_true', 'recession_true_pred', 'recession_false', 'recession_false_pred', 'false_pos_rate', 'false_neg_rate']

# store iteration calculations
iteration_summaries_rf = []

# loop over data
for result in random_forest_results:
    # extract the relevant data
    data = result[1]
    y_true_pred = pd.DataFrame({'y_actual': data['y_true'], 'y_predicted': data['y_pred']})
    
    # create row of data with the calculations
    true_pos = np.sum(y_true_pred['y_actual'] == 1)
    true_neg = np.sum(y_true_pred['y_actual'] == 0)
    pred_pos = np.sum(y_true_pred['y_predicted'] == 1)
    false_pos_rate = np.sum((y_true_pred['y_actual'] == 0) & (y_true_pred['y_predicted'] == 1)) / (np.sum(y_true_pred['y_actual'] == 0))
    false_neg_rate = np.sum((y_true_pred['y_actual'] == 1) & (y_true_pred['y_predicted'] == 0)) / (np.sum(y_true_pred['y_actual'] == 1))

    # create a list of the stats to pass in
    summary_stats = [true_pos, pred_pos, true_neg, len(y_true_pred) - pred_pos, false_pos_rate, false_neg_rate]
    
    # insert lag name
    summary_stats.insert(0, result[0])
    
    # append to result list
    iteration_summaries_rf.append(summary_stats)

# convert to df
complete_summary_stats_rf = pd.DataFrame(iteration_summaries_rf, columns=headers_false_true_summary)

# print results
print(complete_summary_stats_rf)

# export summary stats
complete_summary_stats_rf.to_excel(writer, sheet_name='random-forest-stats', index=False)
# close writer
writer.close()

# create new writer to export the raw predictions
path_prob_raw = '../raw-pred/rf-pred-raw.xlsx'
writer_raw_prob = pd.ExcelWriter(path_prob_raw, engine='openpyxl')
# loop over all the models and get the raw prediction probabilities
# as well as the y_true data for that loop
for lag, res in zip(lags, random_forest_results):
    # get the results
    res_data = res[1]
    # concat into new df
    res_df = pd.DataFrame({'y_true': res_data['y_true'], 'y_pred_raw': res_data['y_pred_raw']})
    # export to new sheet
    res_df.to_excel(writer_raw_prob, sheet_name=f"{lag}_rf", index=False)
# close writer
writer_raw_prob.close()

# export the model for import into another file
# define the export directory
export_dir = "./model-exports/ran-forest-models"
# create directory if doesn't exist
os.makedirs(export_dir, exist_ok=True)
# loop over the models and extract each of them
for lag, model_data in zip(lags, random_forest_results):
    # write the model to binary and export it using pickle
    with open(os.path.join(export_dir, f"ran-forest-model-{lag}-month-lag.pkl"), "wb") as f:
        # select the model and export
        pickle.dump(model_data[1]['best_model'], f)
