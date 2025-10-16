### UNDERSTANDING DATA AND THEIR ENVIRONEMNT - COURSEWORK ###

#Installing packages
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

#Reading the data 
os.chdir('/Users/luke/Documents/University/MSc Data Science/Data and Environment/Coursework')
store = pd.read_csv('Store.csv')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')



## DESCRIBINBG THE DATA ##

#Store - missing observations
store.value_counts() 

#Store - missing values
store_missing_values = store[store.isnull().any(axis=1)]
store_missing_values.isnull().sum()

#Train - missing observations
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
stores = train['Store'].unique()
start_date = '2013-01-01'
end_date = '2015-07-31'
date_range = pd.date_range(start=start_date, end=end_date)
full_index = pd.MultiIndex.from_product([stores, date_range], names=['Store', 'Date'])
full_df = pd.DataFrame(index=full_index).reset_index()
full_df['Date'] = pd.to_datetime(full_df['Date'])
merged_df = full_df.merge(train, on=['Store', 'Date'], how='left')
missing_data = merged_df[merged_df.isnull().any(axis=1)]
missing_by_store = missing_data['Store'].value_counts()
missing_by_date = missing_data['Date'].value_counts()
#percentage of missing values 
len(missing_data)/(len(train)+len(missing_data))

#Train - missing values 
train_missing_values = train[train.isnull().any(axis=1)] #empty 

#Test - missing observations 
list1 = store['Store'].unique()
list2 = test['Store'].unique()
values_not_in_list2 = [x for x in list1 if x not in list2]

#Test - missing values
test.isnull().sum() #11 missing values on Open
test_missing_values = test[test['Open'].isnull()] #store 622

#Checking for consistent data types
store.apply(lambda col: col.map(type).nunique() if col.dtype == 'O' else 1)
train.apply(lambda col: col.map(type).nunique() if col.dtype == 'O' else 1)
test.apply(lambda col: col.map(type).nunique() if col.dtype == 'O' else 1)
store['PromoInterval'].unique()
train['StateHoliday'].unique()


## PRE-PROCESSING ##

## Transformation and Integration

#Merging the store data with the training and test datsets
attributes_train = pd.merge(train, store, on='Store', how='inner')
attributes_test = pd.merge(test, store, on='Store', how='inner')

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

#Promo2 is integrated into a new variable (test set)
attributes_test['Promo2SinceWeek'] = pd.to_numeric(attributes_test['Promo2SinceWeek'], errors='coerce').fillna(-1).astype(int)
attributes_test['Promo2SinceYear'] = pd.to_numeric(attributes_test['Promo2SinceYear'], errors='coerce').fillna(-1).astype(int)
attributes_test['Promo2Since'] = attributes_test.apply(
    lambda row: calculate_first_day_of_week(row['Promo2SinceYear'], row['Promo2SinceWeek']), axis=1)
attributes_test['Promo2Since'] = pd.to_datetime(attributes_test['Promo2Since'], errors='coerce')
attributes_test['Date'] = pd.to_datetime(attributes_test['Date'], errors='coerce')
attributes_test['Promo2'] = attributes_test.apply(
    lambda row: 1 if pd.notnull(row['Promo2Since']) and row['Date'] >= row['Promo2Since'] else 0,
    axis=1)

#Extracting month from the date
attributes_train['Month'] = attributes_train['Date'].dt.month
attributes_test['Month'] = attributes_test['Date'].dt.month


## Plotting time series grpah to show seasonal trends and differences between promo statuses

attributes_train['YearMonth'] = attributes_train['Date'].dt.to_period('M')
monthly_avg_sales = attributes_train.groupby(['YearMonth', 'Promo', 'Promo2'])['Sales'].mean().reset_index()
monthly_avg_sales['YearMonth'] = monthly_avg_sales['YearMonth'].astype(str)
monthly_avg_sales['YearMonth'] = pd.to_datetime(monthly_avg_sales['YearMonth'])
monthly_avg_sales['Promo_Combination'] = monthly_avg_sales.apply(
    lambda row: 'No Promo' if row['Promo'] == 0 and row['Promo2'] == 0 else
                'Promo' if row['Promo'] == 1 and row['Promo2'] == 0 else
                'Promo2' if row['Promo'] == 0 and row['Promo2'] == 1 else
                'Promo and Promo2', axis=1)

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_sales, x='YearMonth', y='Sales', hue='Promo_Combination', marker='o')
plt.xlabel('Date')
plt.ylabel('Average Monthly Sales')
plt.title('Monthly Average Sales for Different Promotion Combinations')
plt.legend(title='Promotion Type')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


## Data Reduction 

#Variables relating to competition distance are removed
attributes_train.drop(columns=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], inplace=True)
attributes_test.drop(columns=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], inplace=True)

#Extra variables relating to 'Promo 2' dropped
attributes_train.drop(columns=['Promo2Since', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], inplace=True)
attributes_test.drop(columns=['Promo2Since', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], inplace=True)

#Customers are removed as a variable
attributes_train = attributes_train.drop(columns=['Customers'])
attributes_test = attributes_test.drop(columns=['Customers'])

#Removing cases where the store is closed and then dropping open as a variable
attributes_train = attributes_train[attributes_train['Open'] != 0]
attributes_test = attributes_test[attributes_test['Open'] != 0]

#Removing sales from attributes_test 
attributes_test.drop(columns=['Sales'], inplace=True)


## Data Cleaning

#Removing cases of zero sales
zero_sales = attributes_train[attributes_train['Sales'] == 0]
attributes_train = attributes_train[attributes_train['Sales'] != 0]

#Filling missing values for store 622
attributes_test.fillna(0, inplace=True)

#Removing Open as a variable 
attributes_train.drop(columns=['Open'], inplace=True)
attributes_test.drop(columns=['Open'], inplace=True)


#Type-Assortment combination on sales
attributes_train['Store_Assortment_Combo'] = attributes_train['StoreType'] + '-' + attributes_train['Assortment']
plt.figure(figsize=(14, 6))
sns.violinplot(x='Store_Assortment_Combo', y='Sales', data=attributes_train, palette='Set3')
plt.title("Sales Distribution for StoreType and Assortment Combinations", fontsize=16)
plt.xlabel("StoreType and Assortment Combinations", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks(rotation=45)
plt.show()



#One-hot ecnoding explanatory variables 
attributes_train = pd.get_dummies(attributes_train, columns=['Month'], prefix='Month', drop_first=True)
attributes_train = pd.get_dummies(attributes_train, columns=['StoreType'], prefix='StoreType', drop_first=True)
attributes_train = pd.get_dummies(attributes_train, columns=['Assortment'], prefix='Assortment', drop_first=True)
attributes_train = pd.get_dummies(attributes_train, columns=['DayOfWeek'], prefix='DayOfWeek', drop_first=False)
attributes_train = attributes_train.drop(columns=['DayOfWeek_2'])
attributes_train['StateHoliday'] = attributes_train['StateHoliday'].astype(str)
attributes_train = pd.get_dummies(attributes_train, columns=['StateHoliday'], prefix='StateHoliday', drop_first=True)

attributes_test = pd.get_dummies(attributes_test, columns=['Month'], prefix='Month', drop_first=True)
attributes_test = pd.get_dummies(attributes_test, columns=['StoreType'], prefix='StoreType', drop_first=True)
attributes_test = pd.get_dummies(attributes_test, columns=['Assortment'], prefix='Assortment', drop_first=True)
attributes_test = pd.get_dummies(attributes_test, columns=['DayOfWeek'], prefix='DayOfWeek', drop_first=False)
attributes_test = attributes_test.drop(columns=['DayOfWeek_2'])
attributes_test['StateHoliday'] = attributes_test['StateHoliday'].astype(str)
attributes_test = pd.get_dummies(attributes_test, columns=['StateHoliday'], prefix='StateHoliday', drop_first=True)


## Matching columns between train and test
missing_cols = set(attributes_train.columns) - set(attributes_test.columns)
for col in missing_cols:
    attributes_test[col] = False
attributes_test = attributes_test[attributes_train.columns]
attributes_test = attributes_test.drop(columns=['Sales'])




## MODEL FITTING ##

#Creating lists for linear regression outputs
rmse_list = []
rmspe_list = []
coefficients_list = []
predictions_list = []

#Creating a dictionary to store individual models 
store_models = {}

#Making a list of feature columns
feature_columns = ['Promo', 'SchoolHoliday',
                   'Promo2', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
                   'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
                   'DayOfWeek_1', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5',
                   'DayOfWeek_6', 'DayOfWeek_7', 'StateHoliday_a',
                   'StateHoliday_b', 'StateHoliday_c']

#Function to calculate RMSPE
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((np.abs(y_true - y_pred)) / y_true) ** 2))

# Train Linear Regression models for each store
for store_id in attributes_train['Store'].unique():
    store_data = attributes_train[attributes_train['Store'] == store_id].copy()
    X_store = store_data[feature_columns]
    y_store = store_data['Sales']
    model = LinearRegression()
    kf = KFold(n_splits=10, shuffle=False)

    rmse_scores = []
    rmspe_scores = []

    #Perform cross-validation
    for train_index, val_index in kf.split(X_store):
        X_train, X_val = X_store.iloc[train_index], X_store.iloc[val_index]
        y_train, y_val = y_store.iloc[train_index], y_store.iloc[val_index]
        
        #Model fit and prediction
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        
        #Calculkate RMSE and RMSPE for the fold 
        rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_scores.append(rmse_fold)
        rmspe_fold = rmspe(y_val, y_val_pred)
        rmspe_scores.append(rmspe_fold)
    
    #Average RMSE and RMSPE across folds and storing values 
    avg_rmse = np.mean(rmse_scores)
    avg_rmspe = np.mean(rmspe_scores)
    rmse_list.append({'Store': store_id, 'RMSE': avg_rmse})
    rmspe_list.append({'Store': store_id, 'RMSPE': avg_rmspe})
    
    #Model is refit to all of a store's data
    model.fit(X_store, y_store)
    store_models[store_id] = model #stored to allow for future prediction
    
    #Storing coefficiecnts 
    coefficients_dict = {'Store': store_id}
    for feature, coef in zip(feature_columns, model.coef_):
        coefficients_dict[feature] = coef
    coefficients_list.append(coefficients_dict)
    

#Create dataframes 
rmse_df = pd.DataFrame(rmse_list)
rmspe_df = pd.DataFrame(rmspe_list)
coefficients_df = pd.DataFrame(coefficients_list)

#Get the average rmse and rmspe across all models 
average_rmse = rmse_df['RMSE'].mean()
average_rmspe = rmspe_df['RMSPE'].mean()

#Average coefficients
average_coefficients = coefficients_df.drop(columns=['Store']).mean().sort_values(ascending=False)

#Looking at distirbution of RMSPE
rmspe_df.describe()


#Plotting coefficients
plt.figure(figsize=(10, 6))
average_coefficients.plot(kind='barh', color='royalblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Average Coefficients Across Stores')
plt.gca().invert_yaxis()  # Largest bars at the top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()



## MAKING PREDICTIONS ##

predictions_list = []

#Make predicition on each store
for store_id in attributes_test['Store'].unique():
    store_data = attributes_test[attributes_test['Store'] == store_id].copy()
    X_store = store_data[feature_columns]
    model = store_models.get(store_id)
    if model:
        store_data['Predicted_Sales'] = model.predict(X_store)
        predictions_list.append(store_data[['Store', 'Date', 'Predicted_Sales']])

#Create dataframe of predictions
predictions = pd.concat(predictions_list, ignore_index=True)

# Display sample predictions
print(predictions.head())




