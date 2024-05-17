# import packages
import pandas as pd
import numpy as np
from scipy.stats import t
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib
import os
import research_roc_utils.roc_utils as ru

# import the data for the lags
summary_path = '[ADD PATH TO GET THE RAW PREDICTIONS]'

# set up lags
lags = [3, 6, 9, 12, 18]

# import the data for each model and lag
def get_lag_data(lag, model, path):
    return pd.read_excel(path, sheet_name=f"{lag}_{model}")

# get the data for each model
xgbc_true_pred = [('xgbc', lag, get_lag_data(lag, 'xgbc', f"{summary_path}/xgbc-pred-raw.xlsx")) for lag in lags]
rf_true_pred = [('rf', lag, get_lag_data(lag, 'rf', f"{summary_path}/rf-pred-raw.xlsx")) for lag in lags]
nn_true_pred = [('nn', lag, get_lag_data(lag, 'nn', f"{summary_path}/nn-pred-raw.xlsx")) for lag in lags]
svc_true_pred = [('svc', lag, get_lag_data(lag, 'svc', f"{summary_path}/svc-pred-raw.xlsx")) for lag in lags]
logit_true_pred = [('logit', lag, get_lag_data(lag, 'logit', f"{summary_path}/logit-pred-raw.xlsx")) for lag in lags]
cm_true_pred = [('cm', lag, get_lag_data(lag, 'cm', f"{summary_path}/cm-pred-raw.xlsx")) for lag in lags]
# concat to all results
all_results = xgbc_true_pred + rf_true_pred + nn_true_pred + svc_true_pred + logit_true_pred + cm_true_pred

# get the auroc helper
roc_auc_func = ru.auroc_non_parametric

# model list
models = ['xgbc', 'rf', 'nn', 'svc', 'logit', 'cm']
# create all unique combinations of models
all_unique_combos = list(combinations(models, 2))
# reverse list to get other combinations as well
# this is so we can do all comparisons for the p vals
total_combos = [(combo[1], combo[0]) for combo in all_unique_combos] + all_unique_combos

# helper to get first match
def get_y_true(results, lag):
    for data in results:
        if data[1] == lag:
            return data[2]['y_true']
        
# function to get the models
def get_model_predictions(model_1, model_2, lag):
    y_true = get_y_true(all_results, lag)
    model_1_y_pred = [data[2]['y_pred_raw'] for data in all_results if data[0] == model_1 and data[1] == lag][0]
    model_2_y_pred = [data[2]['y_pred_raw'] for data in all_results if data[0] == model_2 and data[1] == lag][0]
    
    return y_true, model_1_y_pred, model_2_y_pred

# export for significant plots
sig_res_plts = '[ENTER PATH TO EXPORT THE DELTA DISTRIBUTION PLOTS]'

# H0: There is no difference between Model 1 and Model 2
# H1: Model 2 is better than Model 1
# function to get the p val
def get_boot_p_val(model_1, model_2, lag):
    y_true, model_1_y_pred, model_2_y_pred = get_model_predictions(model_1, model_2, lag)
    p, z = ru.boot_p_val(y_true, model_1_y_pred, model_2_y_pred, n_resamples=50000,
                         score_fun=roc_auc_func, two_tailed=False, seed=42)
    
    return 1 - p, z

# create histogram of p value
def create_hist_plt(p, deltas, model_1, model_2, lag, lower_ci, mean_value):
    
    plt.figure(figsize=(10, 6))
    plt.hist(deltas, color="lightblue", edgecolor="black", bins=30)
    plt.axvline(x=p, color='darkred', linestyle='-', linewidth=2, label=f'P-Value: {p:.5f}')
    plt.axvline(x=mean_value, color='black', linestyle='--', linewidth=2, label=f'Mean Delta: {mean_value:.3f}')
    plt.axvline(x=lower_ci, color='darkblue', linestyle='-', linewidth=2, label=f'Lower Bound: {lower_ci:.3f}')
    plt.xlabel('AUROC Delta')
    plt.ylabel('Count')
    plt.title(f'{model_1} vs {model_2} AUROC Score Comparison at {lag} Month Lag', fontsize=16)
    plt.legend()
    # increase resolution and save
    file_name = f'{model_1}_{model_2}_lag_{lag}_res.png'
    save_path = os.path.join(sig_res_plts, file_name)
    plt.savefig(save_path, dpi=400)
    plt.show()

model_2_better_res = []
p_test_results = []
# collect the p values
for lag in lags:
    for combo in total_combos:
       p, z = get_boot_p_val(combo[0], combo[1], lag)
       
       # switch the signs of z to make it easier to interpret
       # all this is doing is showing the deltas
       # in terms of model 2's auroc
       # i.e. value of 0.05 means model 2's auroc is 5%
       # larger than model 1's
       z_switched = [-delta for delta in z]
       
       mean_value = np.mean(z_switched)
       
       # num resamples * 1 - confidence
       lower_ci = np.percentile(z_switched, 5)
       
       res = {'models': combo, 'p_val': p, 'deltas': z_switched, 'lag': lag, 'mean': mean_value, 'lower_ci': lower_ci}
       p_test_results.append(res)
       if p < 0.05:
           model_2_better_res.append(res)
           create_hist_plt(p, z_switched, res['models'][0], res['models'][1], lag, lower_ci, mean_value)

# store rows data
rows_data = []
# res['models']: tuple of model 1 and model 2
# res['lag']: the lag
# res['p_val']: calculated p value
for res in model_2_better_res:
    row_data = {'Lag': res['lag'], 'Model 1': res['models'][0],
                'Model 2': res['models'][1], 'P-Value': res['p_val'], 'Mean Outperformance': res['mean'], 'Lower Bound': res['lower_ci']}
    rows_data.append(row_data)
# create df from the list of row data
sig_res = pd.DataFrame(rows_data)
# make sure columns in right order
sig_res = sig_res[['Lag', 'Model 1', 'Model 2', 'P-Value', 'Mean Outperformance', 'Lower Bound']]

# export df to excel
p_val_path = '[EXPORT TEST RESULTS TO EXCEL]'
sig_res.to_excel(p_val_path, index=False)

# func to get data for single model
def get_model_data(model_name, lag):
    model_y_pred = [data[2]['y_pred_raw'] for data in all_results if data[0] == model_name and data[1] == lag][0]
    return model_y_pred
        
# create ROC curves across lags
# path to export plots
roc_curves_path = '[EXPORT PATH TO SAVE THE STACKED ROC PLOTS]'
for lag in lags:
    y_true = get_y_true(all_results, lag)
    model_names = models
    model_preds = [get_model_data(model, lag) for model in models]
    # create plot
    plt_obj = ru.stacked_roc_plt(y_true, model_preds, model_names, rand_guess_color='darkred')
    plt_obj.title(f"Stacked ROC Curve for {lag} Month Lag")
    plt_obj.legend(loc="lower right")
    plt_obj.grid(True)
    file_name = f"stacked_roc_curve_lag_{lag}.png"
    save_path = os.path.join(roc_curves_path, file_name)
    plt_obj.savefig(save_path, dpi=300)
    plt_obj.show()
    
# CREATE EXPLANATORY PLOTS
#------------------------#
# not specific to auroc plots

# create plots of the predictions
# vs. the actual data
# import the data to get the optimal threshold
summary_data = pd.read_excel('[GET THE DATA TO FIND THRESHOLDS]')
# directory to export the figures
recession_plt_path = '[PATH TO EXPORT THE CONFUSION MATRIX PLOTS]'

# func to get the optimal threshold
def get_threshold(model, lag):
    threshold = summary_data.loc[(summary_data['lag'] == lag) & (summary_data['model'] == model), 'threshold'].values[0]
    
    return threshold

def predictions_df(y_true, model_pred, threshold, lag):
    
    binary_pred = (model_pred > threshold).astype(int)

    df = pd.DataFrame({'recession': y_true, 'y_pred': binary_pred})
    true_recession = sum(((df['recession'] == 1) & (df['y_pred'] == 1)).astype(int))
    false_pos = sum(((df['recession'] == 0) & (df['y_pred'] == 1)).astype(int))
    false_neg = sum(((df['recession'] == 1) & (df['y_pred'] == 0)).astype(int))
    true_neg = sum((df.sum(axis=1) == 0).astype(int))

    return_df = pd.DataFrame({'true_neg': true_neg, 'true_pos': true_recession,
                              'false_pos': false_pos, 'false_neg': false_neg},
                             index=[lag])
    return return_df
    
def horizontal_bar_plt(model, lags):
    df_res = []
    for lag in lags[::-1]:
        threshold = get_threshold(model, lag)
        y_true = get_y_true(all_results, lag)
        model_pred = get_model_data(model, lag)
        lag_df = predictions_df(y_true, model_pred, threshold, lag)
        df_res.append(lag_df)
    
    # concat to single df
    all_res = pd.concat(df_res)
    # get sum of counts for each lag
    lag_sums = all_res.sum(axis=1)
    # set up % shares for each lag and model
    # we do this so bar spans entire plot for each model
    plt_lags = [str(lag) for lag in list(all_res.index)]
    true_neg = [val / lag_sum if val != 0 else np.nan for val, lag_sum in zip(list(all_res['true_neg']), lag_sums)]
    true_pos = [val / lag_sum if val != 0 else np.nan for val, lag_sum in zip(list(all_res['true_pos']), lag_sums)]
    false_pos = [val / lag_sum if val != 0 else np.nan for val, lag_sum in zip(list(all_res['false_pos']), lag_sums)]
    false_neg = [val / lag_sum if val != 0 else np.nan for val, lag_sum in zip(list(all_res['false_neg']), lag_sums)]

    cmap = matplotlib.colormaps.get_cmap('tab20')
    # linspace: start, stop, intervals
    colors = cmap(np.linspace(0, 1, 4))  # get 4 colors from colormap
   
    _, ax = plt.subplots(figsize=(10, 6))
    
    b1 = ax.barh(plt_lags, true_neg, color=colors[0], label='True No-Recession', height=0.5)
    b2 = ax.barh(plt_lags, true_pos, left=true_neg, color=colors[1], label='True Recession', height=0.5)
    b3 = ax.barh(plt_lags, false_pos, left=np.add(true_neg, true_pos), color=colors[2], label='False Recession', height=0.5)
    b4 = ax.barh(plt_lags, false_neg, left=np.add(false_pos, np.add(true_neg, true_pos)), color=colors[3], label='False No-Recession', height=0.5)
    
    ax.bar_label(b1, label_type='center', labels=list(all_res['true_neg']), fontsize=12)
    ax.bar_label(b2, label_type='center', labels=list(all_res['true_pos']), fontsize=12)
    ax.bar_label(b3, label_type='center', labels=list(all_res['false_pos']), fontsize=12)
    ax.bar_label(b4, label_type='center', labels=list(all_res['false_neg']), fontsize=12)
    
    ax.set_yticks(plt_lags)
    ax.legend(loc=(0.2, -0.15), ncol=2)
    ax.set_title(f"{model} Results Across Lags", size=12, pad=15)
    ax.get_xaxis().set_ticks([])
    ax.set_ylabel("Lag")
    plt.xlim(0, 1)
    file_name = f"summary_plot_{model}.png"
    save_path = os.path.join(recession_plt_path, file_name)
    plt.savefig(save_path, dpi=400)
    plt.show()

for model in models:
    horizontal_bar_plt(model, lags)
