# import starting packages
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import probplot
import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# set up access
fred = Fred(api_key='API_KEY')

# get series for the NBER recession indicator
nber_recession_data = fred.get_series('USREC')

# path to save plots
fig_dir = '[SET UP PATH TO EXPORT FIGURES]'

# set up variable to help make plots
# with recessions shaded in
# get the dates for recessions
recession_periods = []
# start at index 0
i = 0
# we use this while loop to go through the recession date data
# this loop creates tuples of start and end dates we can use to plot
while i < len(nber_recession_data):
    # check if curr period is a recession
    if nber_recession_data[i] == 1:
        # set the start date of the recession
        recession_start = nber_recession_data.index[i]
        # keep iterating to find the end of the recession
        # start at i + 1 and keep going until no longer recession
        j = i + 1
        while j < len(nber_recession_data) and nber_recession_data[j] == 1:
            j += 1
        # once the end of the recession is found we can create recession pair
        # the default behavior of axvspan is to go up to the end date
        # so we need to specify the end date as the month AFTER the recession ends
        # this will it will include the final month of the recession
        # so this is saying the recession start at recession_start and goes
        # up until recession_end
        recession_end = nber_recession_data.index[j]
        # append a tuple to the recession periods list
        recession_periods.append((recession_start, recession_end))
        # set i to the period after the recession
        i = j + 1
    else:
        i += 1

# get the ten year treasury yield
treasury_10_yr = fred.get_series('DGS10').resample('M').mean()
treasury_10_yr /= 100

# the FRED API's 3-month t-bill rate only goes back to the 80s
# due to this, we can call the data from the yfinance API
# the ticker that represents the 3-month t-bill is ^IRX
# unfortunately, monthly data only goes back to the 80s too
# the daily data goes back much further so we will use that
# then take the average of each month
t_bill_ticker = '^IRX'
# earliest available start date is 1960-01-04
t_bill_start_date = '1960-01-04'
# set end date
# get the current date from the datetime
# https://docs.python.org/3/library/datetime.html#datetime.datetime.now
end_date_yf = datetime.now()
# perform some basic formatting
# date needs to be a string
# and a very specific format
formatted_end_date = end_date_yf.strftime('%Y-%m-%d')
# set the interval to daily
t_bill_interval = '1d'
# make api call
daily_t_bill = yf.download(t_bill_ticker, start=t_bill_start_date, end=end_date_yf, interval=t_bill_interval)
# extract the closing data
t_bill_closing = daily_t_bill['Close']
# group by month and get the average for each month
t_bill_monthly_data = t_bill_closing.resample('M').mean()
# convert data from float to %
t_bill_monthly_data /= 100

# check if data is retrieved successfully
if treasury_10_yr is not None and t_bill_monthly_data is not None:
    # combine data into DataFrame
    df = pd.DataFrame({'10Y': treasury_10_yr, '3MO': t_bill_monthly_data})
    
    # calculate spread (10Y - 3MO)
    df['T10Y3MO'] = df['10Y'] - df['3MO']

# SET UP & GET S&P500 DATA
#------------------------#

# unfortunately, the yfinance api doesn't have enough data
# on yfinance, the S&P500 data only goes back to the mid-80s
# therefore we need to use an external data source
# we use the data from datahub
# https://datahub.io/core/s-and-p-500#pandas
# this goes until 2018 and we can get the rest from yfinance
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# import the sp500 data
# adjust file path
raw_sp500_data = pd.read_csv('./sp500_data.csv')

# create a series of the relevant data
sp500_data = pd.Series(raw_sp500_data['SP500'].values, index=raw_sp500_data['Date'])

# extract the last date so we can use it 
# to get the start date for the yfinance api call
raw_sp500_end_date = sp500_data.index[-1]

# we now need to convert the string to a format we can pass
# curr format: '%m/%d/%Y'
# yfinance format: '%Y-%m-%d'
# we first convert to datetime, then back to string
# pd.to_datetime: https://shorturl.at/dfwzP
# strftime: https://www.programiz.com/python-programming/datetime/strftime
# add one month to the end date so that it gets the data starting in following month
# where the datahub dataset ends
# we can us the relativedelta package to manipulate dates
# dateutil: https://dateutil.readthedocs.io/en/stable/relativedelta.html
formatted_start_date = pd.to_datetime(raw_sp500_end_date, format='%m/%d/%Y')
formatted_start_date = formatted_start_date + relativedelta(months=1)
formatted_start_date = formatted_start_date.strftime('%Y-%m-%d')

# set ticker for sp500
ticker_yf = "^GSPC"

# set start date to match other data
start_date_yf = formatted_start_date

# set end date
# we can use the same end_date we used earlier to make the call
# for the t_bill data
# see lines 33-37

# set interval to monthly
interval = "1mo"

# get stock market return data
market_data = yf.download(ticker_yf, start=start_date_yf, end=end_date_yf, interval=interval)

# create series to extract closing prices
market_data = pd.Series(market_data['Close'])

# now that we have the yfinance data, we convert the original index data to DateTime
# to make the api call, we need to pass in the date as a string
# but it returns the data as DateTime
# the format for the date changes so we pass in mixed as the format of the string
# this means it will match string format with a format it recognizes and converts it
# this is sufficient for converting to the same DateTime strucure used by yfinance
sp500_data.index = pd.to_datetime(sp500_data.index, format='mixed')

# concat the market data with other raw data
all_sp500_data = pd.concat([sp500_data, market_data], axis=0, ignore_index=False)

# GET SENTIMENT DATA
#------------------#

# retrieve data for UMCSENT series 
# University of Michigan's Index of Consumer Sentiment
# this is a reliable sentiment tracker which is used as part
# of the LEI index
# http://www.sca.isr.umich.edu
consumer_sentiment_data = fred.get_series('UMCSENT')
# up until 1978-01-01 the data was quarterly
# after that point it is collected monthly
# to account for this we can use the same cubic spline technique
# to decide which order of interpolation to use we can do a visual inspection
# depending on what the data looks like, we can decide on the order
# first step is to only select data after the date where data is collected monthly
# create the timestamp as a datetime object
test_start_date = pd.to_datetime('1978-01-01')
monthly_data_copy = consumer_sentiment_data[consumer_sentiment_data.index >= test_start_date]
# plot the data
plt.figure(figsize=(10,6))
# we can use the axvspan to plot recessions
# this matplotlib method lets you add a span i.e.
# shaded bar across the vertical axis
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
plt.plot(monthly_data_copy, linewidth=3, color="lightblue")
plt.title("Monthly Consumer Sentiment Data (1978 - 2024)")
plt.ylabel("Sentiment Level")
plt.xlim(monthly_data_copy.index[0], monthly_data_copy.index[-1])
plt.grid(True)
# increase resolution and save
file_name = 'sentiment_1978_2024.png'
save_path = os.path.join(fig_dir, file_name)
plt.savefig(save_path, dpi=400)
plt.show()
# we will run the interpolation and see if the graphs look similar
# we can adjust the order and perform futher visual inspection to find the best fit
monthly_consumer_sentiment_data = consumer_sentiment_data.interpolate(method='spline', order=3)
# select the data prior to 1978-01-01 and plot the data
interpolated_quarterly_sentiment = monthly_consumer_sentiment_data[monthly_consumer_sentiment_data.index < test_start_date]
# plot the data
plt.figure(figsize=(10,6))
plt.plot(interpolated_quarterly_sentiment, label="Interpolated Monthly Data", linewidth=3, color="lightblue")
plt.plot(consumer_sentiment_data[consumer_sentiment_data.index < test_start_date], label="Quarterly Consumer Sentiment",
         linewidth=0, marker='o', markersize=3, color='red')
# we can use the axvspan to plot recessions
# this matplotlib method lets you add a span i.e.
# shaded bar across the vertical axis
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
plt.title("Interpolated Quarterly Consumer Sentiment Data")
plt.ylabel("Sentiment Level")
plt.xlim(interpolated_quarterly_sentiment.index[0], interpolated_quarterly_sentiment.index[-1])
plt.legend()
plt.grid(True)
# increase resolution and save
file_name = 'interploated_sentiment.png'
save_path = os.path.join(fig_dir, file_name)
plt.savefig(save_path, dpi=400)
plt.show()
# plot all the data to check
plt.figure(figsize=(10,6))
# we can use the axvspan to plot recessions
# this matplotlib method lets you add a span i.e.
# shaded bar across the vertical axis
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
plt.plot(monthly_consumer_sentiment_data[monthly_consumer_sentiment_data.index > test_start_date], linewidth=2, color="lightblue")
plt.plot(interpolated_quarterly_sentiment, linestyle="-", linewidth=2, color="darkred")
plt.title("Interpolated Monthly Consumer Sentiment Data")
plt.ylabel("Sentiment Level")
plt.xlim(monthly_consumer_sentiment_data.index[0], monthly_consumer_sentiment_data.index[-1])
plt.grid(True)
# increase resolution and save
file_name = 'interploated_sentiment_full.png'
save_path = os.path.join(fig_dir, file_name)
plt.savefig(save_path, dpi=400)
plt.show()
# from the visual inspection it seems like order=3 fits the data best
# right now the data is not in a format which is particularly useful
# we can transform the data to a different scale to measure changes
# in the overall sentiment which is more useful
# to do this, we first need to check the normality of the data

# create probability plot
#-----------------------#
# the way this type of plot works is by sorting the data in ascending order
# for each data point, the corresponding quantile (i.e., the cumulative probability)
# of the theoretical distribution is calculated
# this theoretical data is plotted and then compared with the actual data
# the closer the points hug the line in the middle, the more normal the data is
probplot(monthly_consumer_sentiment_data, dist='norm', plot=plt)
plt.title('Probability Plot of Monthly Sentiment Data')
# increase resolution and save
file_name = 'sentiment_prob_plt.png'
save_path = os.path.join(fig_dir, file_name)
plt.savefig(save_path, dpi=400)
plt.grid(True)
plt.show()
# the data is not totally normally distributed which is not surprising
# however, the data follows a normal distribution pretty closely
# given this, we'll use scaled values as a measure of deviations from the mean
# unlike z-score standardization, which assumes normality, scaled values 
# provide a straightforward way to transform the data into a more interpretable format
# without relying on specific distribution assumptions
# each scaled value represents its position within the range of the original data
# this metric is valuable for capturing sharp rises and falls in sentiment
# which likely correlate with economic recessions
# while normality is assumed, since the data approaches normality
# the logic of the metric makes sense as a way to transform the data to a more useful format
# we are also not making any assumption about the distrubtion of the datas
# i.e. whether or not it is normal affects the transformation we can do
# but is functionaly irrelevant for the analysis
# we're simply transforming the raw values to a standardized scale for easier interpretation
# which doesn't rely on normality, otherwise we could have used something
# like z-score standardization to measure devaitions from the mean
# Min-Max normalization: https://shorturl.at/esyDZ
# sv = (X - Xmin) / (Xmax - Xmin)

# sklearn expects a 2D array as an input
# i.e. a data structure of rows and columns
# right now the data is a series which is incompatible
# so we can transform this into a 2d array
# we can do this by using the .reshape() method from numpy
# by using reshape(1, -1) we are telling numpy to create
# a 2d array with 1 column that holds all the data
# and the -1 is telling nump[y to infer the number of columns
# based on the data structure which in our case is only 1
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
reshaped_monthly_consumer_sentiment_data = monthly_consumer_sentiment_data.values.reshape(-1, 1)
# define the scaler object
scaler = MinMaxScaler()
# transfom original data
scaled_monthly_sentiment_data = scaler.fit_transform(reshaped_monthly_consumer_sentiment_data)
# convert the numpy array back to series so we can add back the index
# by passing in reshape(-1) we are telling numpy to flatten the array back into one dimensional
scaled_monthly_sentiment_data = pd.Series(scaled_monthly_sentiment_data.reshape(-1), 
                                          index=monthly_consumer_sentiment_data.index)
# perform some exploratory analysis
# since scaled values are not directly interpretable
# i.e. a value of 0.5 doesn't necessarily mean the average
# we can look at the average unscaled value and compare it to the scaled
# value to be better guage what the sentiment values mean
raw_sentiment_avg = np.average(monthly_consumer_sentiment_data)
print('Raw Sentiment Avg.: ',raw_sentiment_avg)
scaled_sentiment_avg = np.average(scaled_monthly_sentiment_data)
print('Scaled Sentiment Avg.: ', scaled_sentiment_avg)
# so we can see the avg raw score is about 86
# and the avg scaled score is about 0.6
# so now we can say a scaled score of 0.6 is equivalent to 86
# so anything that falls above 0.6 is above average sentiment
# and anything below 0.6 represents pessemistic sentiment

# SET UP CONSUMER SPENDING VARIBLE
#--------------------------------#

# changes in consumer spending have long been used 
# as a leading recession indicator
# the FED monitors this metric closely
# we can use the FRED to get the PCE dataset
# Personal Consumption Expenditures
# basically a measure of how much US consumers spend
# on durable and non-durable goods
# the spending level is seasonally adjusted
# https://www.investopedia.com/terms/p/pce.asp
# the format of the series is the seasonally adjusted
# annual rate which we then need to adjust for inflation

# create function to calculate the after inflation growth/decline
def find_real_growth(change, infl_rate):
    return (change - infl_rate) / (1 + infl_rate)

# get the retail sales series
consumer_spending_raw = fred.get_series('PCE')
# get inflation data
consumer_goods_inflation = fred.get_series('CPIAUCSL')
# edit the formatting to make it fit
# we want to change the raw spending to the annual change
annual_consumer_spending_change = np.log(consumer_spending_raw / consumer_spending_raw.shift(12))
# the first 12 monthly entries are NaN so we can  safely dropna()
annual_consumer_spending_change = annual_consumer_spending_change.dropna()
# now we do the same thing for the consumer goods inflation index
# the format is a monthly inflation level with 1983-08-01 as ~100
annual_consumer_inflation_change = np.log(consumer_goods_inflation / consumer_goods_inflation.shift(12))
# we can now dropna() here too
annual_consumer_inflation_change = annual_consumer_inflation_change.dropna()
# make a dataframe to line up the date ranges
consumer_spending_inflation_df = pd.DataFrame({'annual_consumer_spending_change': annual_consumer_spending_change,
                                               'annual_consumer_inflation_change': annual_consumer_inflation_change})
# drop the na values so we can find the real change
consumer_spending_inflation_df = consumer_spending_inflation_df.dropna()

# create column to find the real change
consumer_spending_inflation_df['real_consumer_spending_change'] = find_real_growth(consumer_spending_inflation_df['annual_consumer_spending_change'],
                                                                                   consumer_spending_inflation_df['annual_consumer_inflation_change'])
# extract the column for the real change
change_consumer_spending_real = consumer_spending_inflation_df['real_consumer_spending_change']

# set up quarterly basis for consumer spending
quarterly_consumer_spending_change = np.log(consumer_spending_raw / consumer_spending_raw.shift(3))
# drop na values
quarterly_consumer_spending_change = quarterly_consumer_spending_change.dropna()
# now we do the same thing for the consumer goods inflation index
# the format is a monthly inflation level with 1983-08-01 as ~100
quarterly_consumer_inflation_change = np.log(consumer_goods_inflation / consumer_goods_inflation.shift(3))
# we can now dropna() here too
quarterly_consumer_inflation_change = quarterly_consumer_inflation_change.dropna()
# make df of quarterly data
quarterly_consumer_spending_inflation_df = pd.DataFrame({'quarterly_consumer_spending_change': quarterly_consumer_spending_change,
                                               'quarterly_consumer_inflation_change': quarterly_consumer_inflation_change})
# create column to find the real change
consumer_spending_inflation_df['quarterly_real_consumer_spending_change'] = find_real_growth(
                                                                            quarterly_consumer_spending_inflation_df['quarterly_consumer_spending_change'],
                                                                            quarterly_consumer_spending_inflation_df['quarterly_consumer_inflation_change'])
quarterly_spending_change = consumer_spending_inflation_df['quarterly_real_consumer_spending_change']

# GET AND SET UP GDP DATA
#-----------------------#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# the data we recieve is quarterly so we need to rebase this to monthly 
# to achieve this as best as possible, we use spline interpolation   
# this smooths the data between periods to a monthly basis
# this method is also employed in literature which uses FRED API data
# https://www.geeksforgeeks.org/cubic-spline-interpolation/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# get gdp data
# GDPC1 get real inflation adjusted quarterly data
# the data is in 2017 dollars as a base
gdp_data = fred.get_series('GDPC1')
# in this step we are converting the series index to monthly
# this isn't resampling, it is just changing the date format
# this is necessary to perform the subsequent data manipulation
gdp_data.index = gdp_data.index.to_period(freq='M')

# resample to monthly frequency and fill missing months with NaN
# we resample the quarterly data to monthly
# resample requires some sort of chaining method, but using .asfreq()
# without passing anything in it fills the missing monthly values with NaN
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
monthly_gdp_data = gdp_data.resample('M').asfreq()

# convert the index to DateTime
# we need to do this, otherwise the splice method won't work
# right now the index is a period index and not DateTime
# certain pandas methods only work when the index is DateTime
# the method for converting a PeriodIndex to DateTime is to_timestamp()
# https://shorturl.at/KQ468
monthly_gdp_data.index = monthly_gdp_data.index.to_timestamp()

# perform spline interpolation to fill NaN values
# spline interpolation uses polynomials to estimate missing values
# the order property is used to determine the smoothness of the curve
# order=3 means a cubic polynomial function is used (i.e. linear would be order=1)
# this is often used to balance between capturing patterns and avoiding overfitting
monthly_gdp_data_interpolated = monthly_gdp_data.interpolate(method='spline', order=3)

# calculate the date range for the last 20 years
# viewing plot of whole data will make it hard to accurately inspect
# by picking subset we can better see if the data is very different
end_date = monthly_gdp_data_interpolated.index[-1]
start_date = end_date - timedelta(days=365 * 20)

# find the nearest available timestamps in the DataFrame's index
# we do this to set the limits for the y-axis for easier interpretation
# if we didn't do this the gdp line would look like just a flat line
# we need to make this adjustment because the start and end date might not choose
# exact dates available in the index
# for setting the timespan limit this is fine but for the y-axis this is relevant
# we use the asof() method to find the closest available valid date based on the start and end
# this lets us pass in a date and find the closest match
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asof.html
start_date_nearest = monthly_gdp_data_interpolated.index.asof(start_date)
end_date_nearest = monthly_gdp_data_interpolated.index.asof(end_date)

# create a plot of the GDP data for visual inspection
plt.figure(figsize=(10,6))
# we can use the axvspan to plot recessions
# this matplotlib method lets you add a span i.e.
# shaded bar across the vertical axis
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
plt.plot(monthly_gdp_data_interpolated, label='Interpolated Monthly GDP', linewidth=3, color="lightblue")
plt.plot(gdp_data, label='Quarterly Actual', linewidth=0, marker='o', markersize=3, color='red')
plt.title('Interpolated Real Monthly GDP vs. Quarterly Actual')
plt.xlabel('Time')
# set x limits to date range
plt.xlim(start_date, end_date)
# create y limit for min to better see growth/decline
plt.ylim(monthly_gdp_data_interpolated[start_date_nearest], monthly_gdp_data_interpolated[end_date_nearest] + 1000)
plt.ylabel('GDP ($ Billion - 2017 Dollars)')
plt.legend()
# increase resolution and save
file_name = 'gdp_interpolated.png'
save_path = os.path.join(fig_dir, file_name)
plt.savefig(save_path, dpi=400)
plt.legend()
plt.grid(True)
plt.show()

# plot looks good
# looks like the interpolation fit the data well
# we now convert the monthly gdp data to the growth/decline
# we will use continuous log change
sa_real_gdp_change = np.log(monthly_gdp_data_interpolated / monthly_gdp_data_interpolated.shift(6)).dropna()

# SET UP INDEPENDENT VARIABLES
#----------------------------#

# create variable for quarterly return
# calculate log returns for each quarter using numpy log method
# shift function returns the close in the previous row
sp_quarterly_change = np.log(all_sp500_data / all_sp500_data.shift(3)).dropna()

# extract the diff as yield curve
yield_curve = df['T10Y3MO'].dropna()

# create variables of lagged term spread
three_m_lagged_term_spread = df['T10Y3MO'].shift(3)

# create dataframe of all inputs
all_data = pd.DataFrame({'sp_q_change': sp_quarterly_change,
                         'scaled_consumer_sentiment': scaled_monthly_sentiment_data,
                         'yield_curve': yield_curve,
                         '3_month_lagged_yc': three_m_lagged_term_spread,
                         '6_month_gdp_change': sa_real_gdp_change,
                         'ann_change_consumer_spending': change_consumer_spending_real,
                         'nber_recession': nber_recession_data})

# since the data is all monthly, but the dates are different we need to make an adjustment
# some data is collected at the beggining of month and some at the end
# we group it therefore by the month and not the specific day
# we need to add an aggregation function to the groupby
# since the data is already monthly anyways, by adding sum we are not changing any values
# it is esentially rebasing the data to monthly and adding 0 to whatever the value is
# the original monthly values from the unstructured dataframe remains unchanged
all_data = all_data.groupby(all_data.index.to_period('M')).sum()

# find the first row which is not zero for all columns besides the last one
# iloc[] uses indexing so we select all rows, then all columns 
# besides the last one by passing in -1
# we then filter for non-zero values
# != 0 performs element wise operations on each "cell"
# this creates a boolean dataframe where any non-zero value is true
# and any value which is 0 or NaN is considered false
# second, we use all() to find the rows where all values are truthy
# all() essentially checks whether all the elements are truthy or not
# we pass in axis=1 to check column wise
# this is saying: "check each column and see where they are all truthy"
# this returns a series the length of the original dataframe with true or false values
# each row of the returned series represents an index value in our case
# so we are using this series of boolean values to extract the index
# that we want to use as the start of our data, so everything following that index
# https://shorturl.at/bpBL8
# idxmax() is then used to find the row where the index is maxed
# idxmax() returns the maximum value, which in our case is true (1)
# since the values are all either 0 or 1, it returns the first instance
# https://www.geeksforgeeks.org/python-pandas-dataframe-idxmax/
first_non_zero_row = (all_data.iloc[:, :-1] != 0).all(axis=1).idxmax()
# use loc[] to select all the rows and columns after the specific index we calculated
# we can use a slicing operation to select all the rows and columns after this main 
all_data = all_data.loc[first_non_zero_row:]
# now also do the same thing but check the end
# the gdp data is released quarterly we need a full quarter to get full data
# for that reason, as of Mar. 2023 the current querter is not yet over and most recent
# data has not been released so we need to also make sure that we have full gdp data
# and other data that is released quarterly which we need to interpolate
# we do the exact same thing as before, we just now go in the opposite direction
last_non_zero_row = (all_data.iloc[:, :-1] != 0).all(axis=1)[::-1].idxmax()
# now update the data and extract up to that column
all_data = all_data.loc[:last_non_zero_row]

# CHECK FOR MULTICOLLINEARITY IN THE DATA
#---------------------------------------#

# create a copy of the training data
collinearity_check_df = all_data.copy(deep=True)
# drop the target variable
collinearity_check_df = collinearity_check_df.drop('nber_recession', axis=1)
# create correlation matrix
corr_matrix_ind_vars = collinearity_check_df.corr()
# create heatmap of results
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(corr_matrix_ind_vars, cmap='coolwarm')
# add colorbar
plt.colorbar(heatmap)
# add correlation values to the individual boxes
# first loop over each row, then inner loop over
# each column value for that row
# we then use the index values to assign
# text values each box's value
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
for i in range(corr_matrix_ind_vars.shape[0]):
    for j in range(corr_matrix_ind_vars.shape[1]):
        plt.text(j, i, f'{corr_matrix_ind_vars.iloc[i, j]:.2f}', ha='center', va='center', color='white')
# create ticks for x and y columns by first passing in a list of indicies
# then the feature names to match with each index
plt.xticks(range(len(corr_matrix_ind_vars.columns)), corr_matrix_ind_vars.columns, rotation=45)
plt.yticks(range(len(corr_matrix_ind_vars.columns)), corr_matrix_ind_vars.columns)
# add titles and labels
plt.title('Multicollinearity of Independent Variables')
plt.tight_layout()
# increase resolution and save
file_name = 'correlation_matrix_vars.png'
save_path = os.path.join(fig_dir, file_name)
plt.savefig(save_path, dpi=400)
plt.show()

# VIF: Variance Inflation Factor
# https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
# create VIF dataframe
vif_stats = pd.DataFrame()
# get columns for ind vars
vif_stats['feature'] = collinearity_check_df.columns
# loop and calc for each variable
vif_stats['VIF_score'] = [variance_inflation_factor(collinearity_check_df.values, i)
                          for i in range(len(collinearity_check_df.columns))]

# explore the data futher
# goal is to make sure the subset of data is sufficiently representative
# there is only so much that can be done here but work trying
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# check how many recession months are in the dataset as a whole
# this is the recession series from the production dataset
recession_series = all_data['nber_recession']
# this data is taken from the raw series from the FRED API
total_recession_months_all_data = np.sum(nber_recession_data)
print('Total Recession Months - All Data: ', total_recession_months_all_data)

total_recession_months = np.sum(recession_series)
print('\nTotal Recession Months - Production Data: ', total_recession_months)

total_share_of_all_recession_data = np.sum(nber_recession_data) / len(nber_recession_data)
print('\nShare of Total Months - All Data:', total_share_of_all_recession_data)

share_of_dataset_total = total_recession_months / len(recession_series)
print('\nShare of Total Months - Production Data:', share_of_dataset_total)

# create function to find the total amount of recessions
def total_recessions_for_period(data):
    # initialize count
    recession_count = 0
    # iterate through the data points starting from the second month
    # we do this so we have a previous month to compare to
    for i in range(1, len(data)):
        current = data[i]
        prev = data[i - 1]
        # check if the current month is in a recession (current == 1) 
        # and the previous month was not in a recession (prev == 0)
        if current == 1 and prev == 0:
            recession_count += 1
    return recession_count

# call function for entire recession series
total_recessions_raw = total_recessions_for_period(nber_recession_data)
print('\nTotal Recessions for Entire Raw Dataset: ', total_recessions_raw)
total_recessions_production = total_recessions_for_period(recession_series)
print('\nTotal Recessions for Production Dataset: ', total_recessions_production)

# look at how often recessions occur now that we have a total count
recession_frequency_raw = (len(nber_recession_data) / 12) / total_recessions_raw
print('Recession Frequency Raw Data: ', recession_frequency_raw)
recession_frequency_production = (len(recession_series) / 12)/ total_recessions_production
print('Recession Frequency Production Data: ', recession_frequency_production)

# export the data to excel
#------------------------#
# create the path and create writer
path = '[ENTER PATH TO EXPORT DATA]'
all_data.to_excel(path, index=True)
