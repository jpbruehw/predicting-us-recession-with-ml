# import packages
import pandas as pd

# import the sheet
summary_stats = pd.read_excel('[IMPORT THE SUMMARY STATISTICS]')

# set up writer to export
path = '[SET UP THE PATH EXPORT METRIC RANKS]'
writer = pd.ExcelWriter(path, engine='openpyxl')

# set up lags
lags = [3, 6, 9, 12, 18]

# metics to consider for ranking
metrics = ['f1', 'recall', 'precision', 'accuracy', 'roc_auc']

# group by the lag to test at different levels
lag_stats = summary_stats.groupby('lag')

# set up a simple ranking dataframe
# this lists the models in terms of performance
# for each lag based on the metrics collected
#---------------------------------------------#
# setup result container
results = []

# loop over the performance results
for lag in lags:
    # extract the grouped results for lag
    lag_data = lag_stats.get_group(lag)
    # set up dictionary to store ranks
    # this creates list we can use as the key for the 
    lag_result = {'lag': [lag] * len(lag_data)}
    # loop over each metric
    for metric in metrics:
        # sort the data based on the current metric
        sorted_rank = lag_data[['model', metric]].sort_values(by=metric, ascending=False)
        # store the results
        lag_result[f'{metric}_model_ranking'] = sorted_rank['model'].tolist()
        lag_result[f'{metric}_values'] = sorted_rank[metric].tolist()
        # convert to dataframe
        lag_result_df = pd.DataFrame(lag_result)
        # set index as the lag
        lag_result_df = lag_result_df.set_index('lag')
    # append the lag_result to results DataFrame
    results.append(lag_result_df)

# concat the dataframes and export
all_results = pd.concat(results)
all_results.to_excel(writer, sheet_name='all-comparisons')

# export each result to an excel sheet
for lag, result in zip(lags, results):
    # export to excel
    result.to_excel(writer, sheet_name=f"{lag}_ranking_result")

# close writer
writer.close()