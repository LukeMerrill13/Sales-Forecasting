### COURSEWORK PRE-PROCESSING ###

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency


## READING THE DATA ##

os.chdir('/Users/luke/Documents/University/MSc Data Science/Data and Environment/Coursework')
store = pd.read_csv('Store.csv')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


## TRANSFORMATION + INTEGRATION ##

#Merging the datasets together
attributes_train = pd.merge(train, store, on='Store', how='inner')
attributes_test = pd.merge(test, store, on='Store', how='inner')

#Removing cases where the store is closed and then dropping open as a variable
attributes_train = attributes_train[attributes_train['Open'] != 0]
attributes_test = attributes_test[attributes_test['Open'] != 0]

#Sales is kept in its original form as a continuous variable
#Removing cases of zero sales
attributes_train = attributes_train[attributes_train['Sales'] != 0]
#Removing sales from attributes_test 
attributes_test.drop(columns=['Sales'], inplace=True)


#Customers are removed as a variable
attributes_train = attributes_train.drop(columns=['Customers'])
attributes_test = attributes_test.drop(columns=['Customers'])


#Variables relating to competition distance are removed
attributes_train.drop(columns=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], inplace=True)
attributes_test.drop(columns=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], inplace=True)

#Promo2 is integrated into a new variable (train set)
attributes_train['Promo2SinceWeek'] = pd.to_numeric(attributes_train['Promo2SinceWeek'], errors='coerce').fillna(-1).astype(int)
attributes_train['Promo2SinceYear'] = pd.to_numeric(attributes_train['Promo2SinceYear'], errors='coerce').fillna(-1).astype(int)
def calculate_first_day_of_week(year, week):
    try:
        # Use the year and week number to calculate the first day of the week
        return datetime.datetime.strptime(f'{year}-W{int(week)}-1', "%Y-W%U-%w").date()
    except (ValueError, TypeError):
        return None  # Handle invalid or NaN values by returning None
attributes_train['Promo2Since'] = attributes_train.apply(
    lambda row: calculate_first_day_of_week(row['Promo2SinceYear'], row['Promo2SinceWeek']), axis=1)
attributes_train['Promo2Since'] = pd.to_datetime(attributes_train['Promo2Since'], errors='coerce')
attributes_train['Date'] = pd.to_datetime(attributes_train['Date'], errors='coerce')
attributes_train['Promo2'] = attributes_train.apply(
    lambda row: 1 if pd.notnull(row['Promo2Since']) and row['Date'] >= row['Promo2Since'] else 0,
    axis=1)
attributes_train.drop(columns=['Promo2Since', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], inplace=True)

#Promo2 is integrated into a new variable (test set)
attributes_test['Promo2SinceWeek'] = pd.to_numeric(attributes_test['Promo2SinceWeek'], errors='coerce').fillna(-1).astype(int)
attributes_test['Promo2SinceYear'] = pd.to_numeric(attributes_test['Promo2SinceYear'], errors='coerce').fillna(-1).astype(int)
def calculate_first_day_of_week(year, week):
    try:
        # Use the year and week number to calculate the first day of the week
        return datetime.datetime.strptime(f'{year}-W{int(week)}-1', "%Y-W%U-%w").date()
    except (ValueError, TypeError):
        return None  # Handle invalid or NaN values by returning None
attributes_test['Promo2Since'] = attributes_test.apply(
    lambda row: calculate_first_day_of_week(row['Promo2SinceYear'], row['Promo2SinceWeek']), axis=1)
attributes_test['Promo2Since'] = pd.to_datetime(attributes_test['Promo2Since'], errors='coerce')
attributes_test['Date'] = pd.to_datetime(attributes_test['Date'], errors='coerce')
attributes_test['Promo2'] = attributes_test.apply(
    lambda row: 1 if pd.notnull(row['Promo2Since']) and row['Date'] >= row['Promo2Since'] else 0,
    axis=1)
attributes_test.drop(columns=['Promo2Since', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], inplace=True)

#Month extracted from the date
attributes_train['Month'] = attributes_train['Date'].dt.month
attributes_test['Month'] = attributes_test['Date'].dt.month


# #Setting an new composite index for attributes_test so forecasts can be identified
# attributes_test.set_index(['Store', 'Date'], inplace=True)

# #Date and Store are dropped from attributes_train
# attributes_train = attributes_train.drop(columns=['Store'])
# attributes_train = attributes_train.drop(columns=['Date'])




## MISSING VALUES ##

#Finding missing values in the train set 
missing_values_train = attributes_train.isnull().sum()
#Finding missing values in the test set 
missing_values_test = attributes_test[attributes_test.isnull().any(axis=1)]
#edit this line
#store_622 = attributes_test.loc[attributes_test.index.get_level_values('Store') == 622]

#Dropping the open varibale in both datasets
attributes_train.drop(columns=['Open'], inplace=True)
attributes_test.drop(columns=['Open'], inplace=True)




## NOISE REDUCTION ##

#State holiday converted into the same datatype
train['StateHoliday'].unique()
attributes_train['StateHoliday'] = attributes_train['StateHoliday'].astype(str)
attributes_test['StateHoliday'] = attributes_test['StateHoliday'].astype(str)





## DATA REDUCTION ##

#Correlation analysis between explanatory variables (chi-suqared tests)
explanatory_vars = [col for col in attributes_train.columns if col != 'Sales']
p_values = pd.DataFrame(np.zeros((len(explanatory_vars), len(explanatory_vars))), 
                        columns=explanatory_vars, 
                        index=explanatory_vars)
for col1 in explanatory_vars:
    for col2 in explanatory_vars:
        if col1 != col2:
            contingency_table = pd.crosstab(attributes_train[col1], attributes_train[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            p_values.loc[col1, col2] = p
print("Chi-squared Test P-values Matrix (Excluding 'Sales'):")
print(p_values)
#Chi-squared tests all showing dependency but this can't be right




## ENCODING EXPLANATORY VARIABLES ##

attributes_train = pd.get_dummies(attributes_train, columns=['Month'], prefix='Month', drop_first=True)
attributes_train = pd.get_dummies(attributes_train, columns=['StoreType'], prefix='StoreType', drop_first=True)
attributes_train = pd.get_dummies(attributes_train, columns=['Assortment'], prefix='Assortment', drop_first=True)
attributes_train = pd.get_dummies(attributes_train, columns=['DayOfWeek'], prefix='DayOfWeek', drop_first=False)
attributes_train = attributes_train.drop(columns=['DayOfWeek_2'])
attributes_train = pd.get_dummies(attributes_train, columns=['StateHoliday'], prefix='StateHoliday', drop_first=True)
#attributes_train = pd.get_dummies(attributes_train, columns=['SalesCategory'], prefix='SalesCategory', drop_first=True)

attributes_test = pd.get_dummies(attributes_test, columns=['Month'], prefix='Month', drop_first=True)
attributes_test = pd.get_dummies(attributes_test, columns=['StoreType'], prefix='StoreType', drop_first=True)
attributes_test = pd.get_dummies(attributes_test, columns=['Assortment'], prefix='Assortment', drop_first=True)
attributes_test = pd.get_dummies(attributes_test, columns=['DayOfWeek'], prefix='DayOfWeek', drop_first=False)
attributes_test = attributes_test.drop(columns=['DayOfWeek_2'])
attributes_test = pd.get_dummies(attributes_test, columns=['StateHoliday'], prefix='StateHoliday', drop_first=True)
#attributes_test = pd.get_dummies(attributes_test, columns=['SalesCategory'], prefix='SalesCategory', drop_first=True)

#Matching columns between train and test
missing_cols = set(attributes_train.columns) - set(attributes_test.columns)
for col in missing_cols:
    attributes_test[col] = False
attributes_test = attributes_test[attributes_train.columns]
attributes_test = attributes_test.drop(columns=['Sales'])



























