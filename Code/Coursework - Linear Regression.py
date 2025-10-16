import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Lists to store RMSE and RMSPE values, and coefficients
rmse_list = []
rmspe_list = []
coefficients_list = []
predictions_list = []

# Dictionary to store trained models for each store
store_models = {}

# Define feature columns (same as before)
feature_columns = ['Promo', 'SchoolHoliday',
                   'Promo2', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
                   'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12', 'StoreType_b',
                   'StoreType_c', 'StoreType_d', 'Assortment_b', 'Assortment_c',
                   'DayOfWeek_1', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5',
                   'DayOfWeek_6', 'DayOfWeek_7', 'StateHoliday_a',
                   'StateHoliday_b', 'StateHoliday_c']

# Function to calculate RMSPE
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((np.abs(y_true - y_pred)) / y_true) ** 2))

# Train Linear Regression models for each store
for store_id in attributes_train['Store'].unique():
    store_data = attributes_train[attributes_train['Store'] == store_id].copy()
    X_store = store_data[feature_columns]
    y_store = store_data['Sales']
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Set up KFold for 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=False)
    
    # Lists to hold RMSE and RMSPE for each fold
    rmse_scores = []
    rmspe_scores = []

    # Perform cross-validation
    for train_index, val_index in kf.split(X_store):
        X_train, X_val = X_store.iloc[train_index], X_store.iloc[val_index]
        y_train, y_val = y_store.iloc[train_index], y_store.iloc[val_index]
        
        # Train the model on the current fold
        model.fit(X_train, y_train)
        
        # Predict on the validation fold
        y_val_pred = model.predict(X_val)
        
        # Calculate RMSE for this fold
        rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_scores.append(rmse_fold)
        
        # Calculate RMSPE for this fold
        rmspe_fold = rmspe(y_val, y_val_pred)
        rmspe_scores.append(rmspe_fold)
    
    # Calculate average RMSE and RMSPE across all folds
    avg_rmse = np.mean(rmse_scores)
    avg_rmspe = np.mean(rmspe_scores)
    
    # Store RMSE and RMSPE values for each store
    rmse_list.append({'Store': store_id, 'RMSE': avg_rmse})
    rmspe_list.append({'Store': store_id, 'RMSPE': avg_rmspe})
    
    # Train the model on the entire dataset for the store and store coefficients
    model.fit(X_store, y_store)
    store_models[store_id] = model
    
    # Store coefficients (feature importance)
    coefficients_dict = {'Store': store_id}
    for feature, coef in zip(feature_columns, model.coef_):
        coefficients_dict[feature] = coef
    coefficients_list.append(coefficients_dict)
    
    print(f'Store {store_id} Average RMSE from 10-fold CV: {avg_rmse:.2f}')
    print(f'Store {store_id} Average RMSPE from 10-fold CV: {avg_rmspe:.2f}')  # Print RMSPE for each store

# Convert RMSE, RMSPE, and Coefficients into DataFrames
rmse_df = pd.DataFrame(rmse_list)
rmspe_df = pd.DataFrame(rmspe_list)
coefficients_df = pd.DataFrame(coefficients_list)

# Save RMSE, RMSPE, and Coefficients for analysis
rmse_df.to_csv('linear_rmse_per_store_cv.csv', index=False)
rmspe_df.to_csv('linear_rmspe_per_store_cv.csv', index=False)
coefficients_df.to_csv('linear_coefficients_cv.csv', index=False)

print("RMSE, RMSPE, and Coefficients saved!")

# Compute the average RMSE across all stores
average_rmse = rmse_df['RMSE'].mean()
print(f'Average RMSE across all stores: {average_rmse:.2f}')

# Compute the average RMSPE across all stores
average_rmspe = rmspe_df['RMSPE'].mean()
print(f'Average RMSPE across all stores: {average_rmspe:.2f}')


predictions_list = []

# Iterate through each store and make predictions
for store_id in attributes_test['Store'].unique():
    store_data = attributes_test[attributes_test['Store'] == store_id].copy()
    X_store = store_data[feature_columns]  # Extract features
    
    # Retrieve trained model for this store
    model = store_models.get(store_id)
    
    if model:  # Ensure model exists
        store_data['Predicted_Sales'] = model.predict(X_store)  # Make predictions
        predictions_list.append(store_data[['Store', 'Date', 'Predicted_Sales']])  # Append relevant columns

# Combine all predictions into a single DataFrame
predictions_df = pd.concat(predictions_list, ignore_index=True)

# Save predictions to CSV
predictions_df.to_csv('test_predictions.csv', index=False)

# Display sample predictions
print(predictions_df.head())





