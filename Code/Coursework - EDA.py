import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


#Reading the data 
os.chdir('/Users/luke/Documents/University/MSc Data Science/Data and Environment/Coursework')
store = pd.read_csv('Store.csv')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

attributes_train = pd.merge(train, store, on='Store', how='inner')
attributes_test = pd.merge(test, store, on='Store', how='inner')
#Removing cases where the store is closed
attributes_train = attributes_train[attributes_train['Open'] != 0]


## UNIVARIATE EDA and Against Sales##

#Sales historgram 
plt.figure(figsize=(10, 6))
sns.histplot(attributes_train['Sales'], bins=30, kde=True, color='blue')
plt.title('Distribution of Sales', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

#Sales statistcal distribution
attributes_train['Sales'].describe()

#Number of days with 0 sales
zero_sales = attributes_train[attributes_train['Sales'] == 0]
#no days with positive sales when closed
len(attributes_train[(attributes_train['Open'] == 0) & (attributes_train['Sales'] != 0)])

#How do sales move over time?
attributes_train['Date'] = pd.to_datetime(attributes_train['Date'])
daily_sales = attributes_train.groupby('Date')['Sales'].sum()
plt.figure(figsize=(14, 6))
plt.bar(daily_sales.index, daily_sales.values, color='blue', alpha=0.7)
plt.title('Total Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sum(attributes_train['Sales'] == 0)


#Customers historgram 
plt.figure(figsize=(10, 6))
sns.histplot(attributes_train['Customers'], bins=30, kde=True, color='blue')
plt.title('Distribution of Customers', fontsize=16)
plt.xlabel('Customers', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

#Customers statistcal distribution
attributes_train['Customers'].describe()

#Days with 0 customers
zero_customers = attributes_train[attributes_train['Customers'] == 0]
len(zero_customers['Store'].unique())

#what is the average spend per store?
attributes_train['AvgSpendPerCustomer'] = attributes_train['Sales'] / attributes_train['Customers']
attributes_train['AvgSpendPerCustomer'] = attributes_train['AvgSpendPerCustomer'].replace([float('inf'), -float('inf'), None], 0)
avg_spend_per_store = attributes_train.groupby('Store')['AvgSpendPerCustomer'].mean()
np.mean(avg_spend_per_store)

#How does the number of custuomers affect sales? (scatter)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=attributes_train, x='Customers', y='Sales', alpha=0.5, color='blue')
plt.title('Scatter Plot of Sales vs Customers')
plt.xlabel('Customers')
plt.ylabel('Sales')
plt.show()

#Percentage of days open
attributes_train['Open'].mean()

#Percentage of days with a promo applied
attributes_train['Promo'].mean()

#Which days have more promotions
attributes_train['DayOfWeek'] = attributes_train['Date'].dt.day_name()
pd.crosstab(attributes_train['Promo'], attributes_train['DayOfWeek'])

#Promotion over time
plt.figure(figsize=(14, 6))
plt.bar(promo_day.index, promo_day.values, color='blue', alpha=0.7)
plt.title('Number of Promotions Per Day Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Promotions', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

#average length of a promotion period 
promo_binary = (promo_day > 0).astype(int)
promo_changes = np.diff(promo_binary, prepend=0, append=0)
start_indices = np.where(promo_changes == 1)[0] 
end_indices = np.where(promo_changes == -1)[0]   
promo_lengths = end_indices - start_indices
average_promo_length = np.mean(promo_lengths) if len(promo_lengths) > 0 else 0

#Comparing sales with and without promotion
attributes_train['Date'] = pd.to_datetime(attributes_train['Date'])
promo_day.index = pd.to_datetime(promo_day.index)  # Ensure promo_day index is datetime
daily_sales = attributes_train.groupby('Date')['Sales'].sum()
bar_colors = ['red' if promo_day.get(date, 0) > 0 else 'blue' for date in daily_sales.index]
plt.figure(figsize=(14, 6))
plt.bar(daily_sales.index, daily_sales.values, color=bar_colors, alpha=0.7)
plt.title('Total Sales Over Time (Promo Days in Red)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='No Promotion'),
                   Patch(facecolor='red', label='Promotion Applied')]
plt.legend(handles=legend_elements)
plt.show()

#percentage of days with promotion across store types
total_days_by_store = attributes_train.groupby('StoreType')['Date'].nunique()
promo_days_by_store = attributes_train[attributes_train['Promo'] == 1].groupby('StoreType')['Date'].nunique()
promo_percentage_by_store = (promo_days_by_store / total_days_by_store) * 100
print(promo_percentage_by_store)


#Percentage of days affected by school holiday
attributes_train['SchoolHoliday'].mean()

#School holiday box plot comparison
attributes_train['SchoolHoliday'] = attributes_train['SchoolHoliday'].astype(int)
plt.figure(figsize=(8, 6))
sns.boxplot(x=attributes_train['SchoolHoliday'], y=attributes_train['Sales'])
plt.title("Sales Distribution on School Holidays vs. Regular Days", fontsize=14)
plt.xlabel("School Holiday (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks([0, 1], ["No School Holiday", "School Holiday"])  # Custom x-axis labels
plt.show()

#Percentage of days affectd by state holidays
attributes_train['StateHoliday'] = attributes_train['StateHoliday'].astype(str)
attributes_train['StateHoliday'].value_counts()
len(attributes_train[attributes_train['StateHoliday'] != '0'])/len(attributes_train)

#Violin plot for state holiday sales
sns.violinplot(x="StateHoliday", y="Sales", data=attributes_train, palette="Set3")
plt.title("Sales by Day of the Week")
plt.ylabel("Sales")
plt.show()

#Promo and state holiday
promo_and_state_holiday = attributes_train[(attributes_train['Promo'] > 0) & (attributes_train['StateHoliday'] != '0')]
promo_and_state_holiday_days = promo_and_state_holiday[['Date', 'Promo', 'StateHoliday']]
print(promo_and_state_holiday_days)

#sales by holiday types
pd.pivot_table(attributes_train, values='Sales', 
                                         index='StateHoliday', columns='SchoolHoliday', 
                                         aggfunc='mean')

#looking into easter sales
easter_sales = attributes_train[attributes_train['StateHoliday'] == 'b']['Sales']
easter_sales.describe()

plt.figure(figsize=(10, 6))
sns.histplot(easter_sales, bins=30, kde=True, color='blue')
plt.title('Distribution of Sales', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()


#Competition Distance histogram 
plt.figure(figsize=(10, 6))
sns.histplot(store['CompetitionDistance'], bins=30, kde=True, color='blue')
plt.title('Distribution of Competition Distance', fontsize=16)
plt.xlabel('Competition Distance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

#Competition Distance statistics
store['CompetitionDistance'].describe()

#Competition Since exploration
store['CompetitionSince'] = pd.to_datetime(
    store['CompetitionOpenSinceYear'].fillna(2020).astype(int).astype(str) + '-' +
    store['CompetitionOpenSinceMonth'].fillna(1).astype(int).astype(str) + '-01', 
    format='%Y-%m-%d', 
    errors='coerce'
)
reference_date = pd.to_datetime('2013-01-01')
before_2013 = store[store['CompetitionSince'] < reference_date]
percentage_before_2013 = (len(before_2013) / len(store)) * 100
100 - percentage_before_2013

#Promo 2 percentage
store['Promo2'].mean()

# Promo 2 (change Promo2 so it only says 1 when 'Date' is more recent than 'Promo2Since')
attributes_train['Promo2SinceWeek'] = pd.to_numeric(attributes_train['Promo2SinceWeek'], errors='coerce').fillna(-1).astype(int)
attributes_train['Promo2SinceYear'] = pd.to_numeric(attributes_train['Promo2SinceYear'], errors='coerce').fillna(-1).astype(int)
def calculate_first_day_of_week(year, week):
    try:
        # Use the year and week number to calculate the first day of the week
        return datetime.datetime.strptime(f'{year}-W{int(week)}-1', "%Y-W%U-%w").date()
    except (ValueError, TypeError):
        return None  # Handle invalid or NaN values by returning None
attributes_train['Promo2Since'] = attributes_train.apply(
    lambda row: calculate_first_day_of_week(row['Promo2SinceYear'], row['Promo2SinceWeek']), axis=1
)
attributes_train['Promo2Since'] = pd.to_datetime(attributes_train['Promo2Since'], errors='coerce')
attributes_train['Date'] = pd.to_datetime(attributes_train['Date'], errors='coerce')
attributes_train['Promo2'] = attributes_train.apply(
    lambda row: 1 if pd.notnull(row['Promo2Since']) and row['Date'] >= row['Promo2Since'] else 0,
    axis=1
)

#Box plot of promo2 v no promo2
plt.figure(figsize=(8, 6))
sns.boxplot(x=attributes_train['Promo2'], y=attributes_train['Sales'])
plt.title("Sales Distribution: Promo2 vs No Promo2", fontsize=14)
plt.xlabel("Promo2 (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks([0, 1], ["No Promo2", "Promo2"]) 
plt.show()

#Boxplot to show how combinations of promos affect sales
attributes_train['Promo_Combo'] = attributes_train['Promo'].astype(str) + '-' + attributes_train['Promo2'].astype(str)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Promo_Combo', y='Sales', data=attributes_train, palette='Set2')
plt.title("Sales Distribution Across Promo and Promo2 Combinations", fontsize=16)
plt.xlabel("Promo and Promo2 Combinations", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.show()

#promo2 and storetype
pd.crosstab(store['Promo2'], store['StoreType'])

#creating cycle variable in store
def extract_promo_cycles(promo_string):
    if pd.isna(promo_string):  # Handle NaN values
        return [pd.NaT] * 4  # Return 4 NaT values for missing data
    
    months = promo_string.split(',')  # Split the months
    months = [month_mapping[m] for m in months]  # Convert month names to numbers
    
    # Convert to datetime format (1st day of the given month, assuming any year e.g., 2024)
    dates = [pd.to_datetime(f'2024-{m}-01') for m in months]
    
    # Ensure 4 cycles are returned, filling missing cycles with NaT
    while len(dates) < 4:
        dates.append(pd.NaT)
    
    return dates
store[['Cycle1', 'Cycle2', 'Cycle3', 'Cycle4']] = store['PromoInterval'].apply(lambda x: pd.Series(extract_promo_cycles(x)))

#average monthly sales for different promo intevals 
attributes_train['YearMonth'] = attributes_train['Date'].dt.to_period('M')
attributes_train['PromoCategory'] = attributes_train['PromoInterval'].fillna('No Promo')
monthly_sales = attributes_train.groupby(['YearMonth', 'PromoCategory'])['Sales'].mean().reset_index()
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
monthly_sales['YearMonth'] = pd.to_datetime(monthly_sales['YearMonth'])
all_months = pd.date_range(start=monthly_sales['YearMonth'].min(), 
                           end=monthly_sales['YearMonth'].max(), 
                           freq='MS')
plt.figure(figsize=(14, 7))
for promo_group in monthly_sales['PromoCategory'].unique():
    subset = monthly_sales[monthly_sales['PromoCategory'] == promo_group]
    plt.plot(subset['YearMonth'], subset['Sales'], label=promo_group)
plt.xlabel('Date (Year-Month)')
plt.ylabel('Average Monthly Sales')
plt.title('Average Monthly Sales by PromoInterval Category')
plt.xticks(all_months, all_months.strftime('%Y-%m'), rotation=45)
plt.legend(title="PromoInterval")
plt.grid(True)
plt.show()

#StoreType
store['StoreType'].value_counts()

#Store type violin plot
sns.violinplot(x="StoreType", y="Sales", data=attributes_train, palette="Set2")
plt.title("Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Sales")
plt.show()

#Assortment
store['Assortment'].value_counts()

#Violin plot by Assortment
sns.violinplot(x="Assortment", y="Sales", data=attributes_train, palette="Set3")
plt.title("Sales by Assortment")
plt.xlabel("Assortment")
plt.ylabel("Sales")
plt.show()

#Combination of store type and assortmnet
pd.crosstab(store['StoreType'], store['Assortment'])

#Type-Assortment combination on sales
attributes_train['Store_Assortment_Combo'] = attributes_train['StoreType'] + '-' + attributes_train['Assortment']
plt.figure(figsize=(14, 6))
sns.violinplot(x='Store_Assortment_Combo', y='Sales', data=attributes_train, palette='Set3')
plt.title("Sales Distribution for StoreType and Assortment Combinations", fontsize=16)
plt.xlabel("StoreType and Assortment Combinations", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks(rotation=45)
plt.show()

#Type-Assortment combination and openning days
attributes_train['Store_Assortment_Combo'] = attributes_train['StoreType'] + '-' + attributes_train['Assortment']
contingency_table = pd.crosstab(attributes_train['StoreType'], attributes_train['DayOfWeek'])
print(contingency_table)

#StoreType average Sales across different days
contingency_table_avg_sales = attributes_train.groupby(['StoreType', 'DayOfWeek'])['Sales'].mean().unstack()
print(contingency_table_avg_sales)

#Average monthly sales by store and assortment types
attributes_train['Date'] = pd.to_datetime(attributes_train['Date'])
attributes_train['YearMonth'] = attributes_train['Date'].dt.to_period('M')
attributes_train['Store_Assortment'] = attributes_train['StoreType'] + '-' + attributes_train['Assortment']
monthly_sales = attributes_train.groupby(['YearMonth', 'Store_Assortment'])['Sales'].mean().reset_index()
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
monthly_sales['YearMonth'] = pd.to_datetime(monthly_sales['YearMonth'])
all_months = pd.date_range(start=monthly_sales['YearMonth'].min(), 
                           end=monthly_sales['YearMonth'].max(), 
                           freq='MS')
plt.figure(figsize=(14, 7))
for store_group in monthly_sales['Store_Assortment'].unique():
    subset = monthly_sales[monthly_sales['Store_Assortment'] == store_group]
    plt.plot(subset['YearMonth'], subset['Sales'], label=store_group)
plt.xlabel('Date (Year-Month)')
plt.ylabel('Average Monthly Sales')
plt.title('Average Monthly Sales by Store-Assortment Combination')
plt.xticks(all_months, all_months.strftime('%Y-%m'), rotation=45)
plt.legend(title="Store-Assortment")
plt.grid(True)
plt.show()

#Violin plot by day
sns.violinplot(x="DayOfWeek", y="Sales", data=attributes_train, palette="Set3")
plt.title("Sales by Day of the Week")
plt.ylabel("Sales")
plt.show()

#Bar chart for avaerage sales per day of the week by stpre type
avg_sales_by_store_type = attributes_train.groupby(['DayOfWeek', 'StoreType'])['Sales'].mean().unstack()
plt.figure(figsize=(10, 6))
avg_sales_by_store_type.plot(kind='bar', width=0.8, figsize=(10, 6))
plt.title("Average Sales Per Day of the Week (Grouped by Store Type)", fontsize=16)
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Average Sales", fontsize=12)
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.legend(title="Store Type", title_fontsize='13', fontsize='11')
plt.show()

#Violin plot by month
sns.violinplot(x="Month", y="Sales", data=attributes_train, palette="Set3")
plt.title("Sales by Month")
plt.ylabel("Sales")
plt.show()

#Bar chart for avaerage sales per month of the year
attributes_train['Month'] = attributes_train['Date'].dt.month
avg_sales_by_month_and_store = attributes_train.groupby(['Month', 'StoreType'])['Sales'].mean().unstack()
plt.figure(figsize=(12, 6))
avg_sales_by_month_and_store.plot(kind='bar', width=0.8, figsize=(12, 6))
plt.title("Average Sales Per Month (Grouped by Store Type)", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Sales", fontsize=12)
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
plt.legend(title="Store Type", title_fontsize='13', fontsize='11')
plt.show()


# Comp Since (only 51% of values start before 01/01/2013 so we shouldn't use this figure)
attributes_train['CompetitionSince'] = pd.to_datetime(
    attributes_train['CompetitionOpenSinceYear'].fillna(2020).astype(int).astype(str) + '-' +
    attributes_train['CompetitionOpenSinceMonth'].fillna(1).astype(int).astype(str) + '-01', 
    format='%Y-%m-%d', 
    errors='coerce'
)
reference_date = pd.to_datetime('2013-01-01')
before_2013 = attributes_train[attributes_train['CompetitionSince'] < reference_date]
percentage_before_2013 = (len(before_2013) / len(attributes_train)) * 100







## MISSING VALUES ##

# finding missing rows in the train dataset
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







#Competition Distance
attributes_train['Log_CompetitionDistance'] = np.log(attributes_train['CompetitionDistance'].replace(0, np.nan))
plt.figure(figsize=(10, 6))
sns.kdeplot(
    x='Sales', 
    y='Log_CompetitionDistance', 
    data=attributes_train, 
    cmap='Blues', 
    fill=True, 
    thresh=0, 
    levels=10)
plt.title('2D KDE between Sales and Log of CompetitionDistance', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Log(CompetitionDistance)', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(attributes_train['CompetitionDistance'], attributes_train['Sales'], alpha=0.5, color='blue', s=10)
plt.title('Scatter Plot between Sales and CompetitionDistance', fontsize=16)
plt.xlabel('CompetitionDistance', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.tight_layout()
plt.show()


#Competition Since



#Promo2 






#Month


#Promo
promo_1_sales = attributes_train[attributes_train['Promo'] == 1]['Sales']
promo_0_sales = attributes_train[attributes_train['Promo'] == 0]['Sales']
plt.figure(figsize=(10, 6))
sns.histplot(promo_1_sales, color='purple', label='Promo == 1', stat='density', bins=50)
sns.histplot(promo_0_sales, color='pink', label='Promo == 0', stat='density', bins=50)
plt.title('Sales Density by Promo Status', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
promo_1_sales.mean()
promo_0_sales.mean()

promo_1 = attributes_train[attributes_train['Promo'] == 1]
promo_0 = attributes_train[attributes_train['Promo'] == 0]
avg_sales_promo_1 = promo_1.groupby('Date')['Sales'].mean().reset_index()
avg_sales_promo_0 = promo_0.groupby('Date')['Sales'].mean().reset_index()
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Sales', data=avg_sales_promo_1, label='Promo = 1', color='blue')
sns.lineplot(x='Date', y='Sales', data=avg_sales_promo_0, label='Promo = 0', color='red')
plt.title('Average Sales per Day for Promo = 1 and Promo = 0', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)
plt.legend(title='Promo Status', fontsize=10)
plt.tight_layout()
plt.show()

no_promo = attributes_train[(attributes_train['Promo'] == 0) & (attributes_train['Promo2'] == 0)]
promo_1 = attributes_train[(attributes_train['Promo'] == 1) & (attributes_train['Promo2'] == 0)]
promo_2 = attributes_train[(attributes_train['Promo'] == 0) & (attributes_train['Promo2'] == 1)]
promo_12 = attributes_train[(attributes_train['Promo'] == 1) & (attributes_train['Promo2'] == 1)]
avg_sales_no_promo = (no_promo.groupby(no_promo['Date'].dt.to_period('M'))['Sales'].mean().reset_index())
avg_sales_promo_1 = (promo_1.groupby(promo_1['Date'].dt.to_period('M'))['Sales'].mean().reset_index())
avg_sales_promo_2= (promo_2.groupby(promo_2['Date'].dt.to_period('M'))['Sales'].mean().reset_index())
avg_sales_promo_12 = (promo_12.groupby(promo_12['Date'].dt.to_period('M'))['Sales'].mean().reset_index())

avg_sales_no_promo['Month'] = avg_sales_no_promo['Date'].dt.to_timestamp()
avg_sales_promo_1['Month'] = avg_sales_promo_1['Date'].dt.to_timestamp()
avg_sales_promo_2['Month'] = avg_sales_promo_2['Date'].dt.to_timestamp()
avg_sales_promo_12['Month'] = avg_sales_promo_12['Date'].dt.to_timestamp()

plt.figure(figsize=(14, 7))
sns.lineplot(x='Month', y='Sales', data=avg_sales_no_promo, label='No Promo', color='red')
sns.lineplot(x='Month', y='Sales', data=avg_sales_promo_1, label='Only Promo 1', color='blue')
sns.lineplot(x='Month', y='Sales', data=avg_sales_promo_2, label='Only Promo 2', color='purple')
sns.lineplot(x='Month', y='Sales', data=avg_sales_promo_12, label='Promo 1 and 2', color='pink')
plt.title('Average Sales per Month Based on Promos', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)
plt.legend(title='Promo Status', fontsize=10)
plt.tight_layout()
plt.show()





#State Holoday
sns.violinplot(x="StateHoliday", y="Sales", data=attributes_train, palette="Set3")
plt.title("Sales by State Holioday")
plt.ylabel("Sales")
plt.show()


#School Holiday
holiday_1_sales = attributes_train[attributes_train['SchoolHoliday'] == 1]['Sales']
holiday_0_sales = attributes_train[attributes_train['SchoolHoliday'] == 0]['Sales']
plt.figure(figsize=(10, 6))
sns.histplot(promo_1_sales, color='purple', label='Shchool Holiday == 1', stat='density', bins=50)
sns.histplot(promo_0_sales, color='pink', label='School Holiday == 0', stat='density', bins=50)
plt.title('Sales Density by School Holiday', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
holiday_1_sales.mean()
holiday_0_sales.mean()



## MULTIVRIATE EDA ##

#Combination of assortment and store type
pd.crosstab(attributes_train['StoreType'], attributes_train['Assortment'])
attributes_train['StoreType_Assortment'] = attributes_train['StoreType'] + "_" + attributes_train['Assortment']
plt.figure(figsize=(12, 6))
sns.violinplot(x='StoreType_Assortment', y='Sales', data=attributes_train, palette='muted')
plt.title('Sales Distribution by StoreType and Assortment Combinations', fontsize=16)
plt.xlabel('StoreType and Assortment Combination', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()






#Transformation 

plt.scatter(attributes_train["CompetitionDistance"], attributes_train["Sales"], color="blue", alpha=0.7)
plt.title("Scatter Plot between Variable_X and Variable_Y")
plt.xlabel("Variable_X")
plt.ylabel("Variable_Y")
plt.grid(True)
plt.show()



plt.hist(attributes_train["Sales"], bins=50, color="purple", edgecolor="black", alpha=0.7)
plt.title("Distribution of Sales")
plt.xlabel("Turnover in a day")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


plt.hist(store["CompetitionDistance"], bins=30, color="cyan", edgecolor="black", alpha=0.7)
plt.title("Distance to Nearest Competition")
plt.xlabel("Distance (metres)")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


plt.hist(np.log(store["CompetitionDistance"]), bins=30, color="cyan", edgecolor="black", alpha=0.7)
plt.title("Log Distance to Nearest Competition")
plt.xlabel("Log Distance (metres)")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


sns.violinplot(x="StoreType", y="Sales", data=attributes_train, palette="Set2")

plt.title("Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Sales")
plt.show()

sns.violinplot(x="DayOfWeek", y="Sales", data=attributes_train, palette="Set3")

plt.title("Sales by Day of the Week")
plt.ylabel("Sales")
plt.show()






np.mean(train['Promo'])

attributes_train.corr()


#Normalization 

store['Promo2'].unique()



y_column = 'Sales'

# Create scatter plots for each column except 'Sales'
for column in attributes_train.columns:
    if column != 'Sales':
        plt.figure(figsize=(6, 4))
        plt.scatter([column], attributes_train['Sales'], alpha=0.7)
        plt.title(f'Scatter Plot: {column} vs Sales')
        plt.xlabel(column)
        plt.ylabel('Sales')
        plt.grid(True)
        plt.show()
    

attributes_train['Sales']
attributes_train.columns


plt.figure(figsize=(6, 4))
plt.scatter(attributes_train['Customers'], attributes_train['Sales'], alpha=0.7)


#Integration

attributes_train["Sales"].corr(attributes_train['CompetitionOpenSinceMonth'].isnull().astype(int), method="pearson")
len(store) - store['Promo2'].sum()


#Visualisation




ct = pd.crosstab(attributes_train['SalesCategory'], attributes_train['Store_Assortment_Combo'])


import scipy.stats as stats
contingency_table = pd.crosstab(attributes_train['SalesCategory'], attributes_train['Store_Assortment_Combo'])

# Step 2: Perform the Chi-Square Test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Output the results of the Chi-Square test
print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")
n = contingency_table.sum().sum()  # Total number of observations
k = len(contingency_table)         # Number of rows (SalesCategory levels)
r = len(contingency_table.columns) # Number of columns (Store_Assortment_Combo levels)

cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))

# Output Cramér's V
print(f"Cramér's V: {cramers_v}")



import pandas as pd

# Assuming 'attributes_train' contains the 'Promo2', 'Sales', and 'Store' columns

# Step 1: Identify stores with always zero Promo2
always_zero_promo2 = attributes_train.groupby('Store')['Promo2'].nunique() == 1
stores_with_always_zero_promo2 = always_zero_promo2[always_zero_promo2].index

# Step 2: Identify stores with mixed Promo2 values (both 0 and 1)
stores_with_mixed_promo2 = always_zero_promo2[~always_zero_promo2].index

# Step 3: Filter the dataset based on the store types
stores_with_always_zero_promo2_data = attributes_train[attributes_train['Store'].isin(stores_with_always_zero_promo2)]

# Filter the stores with mixed Promo2 and for dates where Promo2 is 0
stores_with_mixed_promo2_data = attributes_train[(attributes_train['Store'].isin(stores_with_mixed_promo2)) & (attributes_train['Promo2'] == 0)]

# Step 4: Calculate the average sales for each group
avg_sales_always_zero = stores_with_always_zero_promo2_data['Sales'].mean()
avg_sales_mixed_promo2_zero = stores_with_mixed_promo2_data['Sales'].mean()

# Output the results
print(f"Average Sales for Stores with Always Zero Promo2: {avg_sales_always_zero}")
print(f"Average Sales for Stores with Mixed Promo2 Values (Promo2 = 0): {avg_sales_mixed_promo2_zero}")


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'stores_with_always_zero_promo2_data' and 'stores_with_mixed_promo2_data' are already defined

# Step 1: Convert the 'Date' column to datetime format
stores_with_always_zero_promo2_data['Date'] = pd.to_datetime(stores_with_always_zero_promo2_data['Date'])
stores_with_mixed_promo2_data['Date'] = pd.to_datetime(stores_with_mixed_promo2_data['Date'])

# Step 2: Extract the year and month from the 'Date' column
stores_with_always_zero_promo2_data['YearMonth'] = stores_with_always_zero_promo2_data['Date'].dt.to_period('M')
stores_with_mixed_promo2_data['YearMonth'] = stores_with_mixed_promo2_data['Date'].dt.to_period('M')

# Step 3: Group by 'YearMonth' and calculate the average sales per month for both datasets
avg_sales_always_zero = stores_with_always_zero_promo2_data.groupby('YearMonth')['Sales'].mean()
avg_sales_mixed_promo2_zero = stores_with_mixed_promo2_data.groupby('YearMonth')['Sales'].mean()

# Step 4: Plot the line graph
plt.figure(figsize=(12, 6))
plt.plot(avg_sales_always_zero.index.astype(str), avg_sales_always_zero, label="Stores with Always Zero Promo2", color='blue')
plt.plot(avg_sales_mixed_promo2_zero.index.astype(str), avg_sales_mixed_promo2_zero, label="Stores with Mixed Promo2 (Promo2=0)", color='red')

# Step 5: Customize the plot
plt.xlabel('Time (Month)', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)
plt.title('Average Sales per Month for Different Store Groups', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()




attributes_train['YearMonth'] = attributes_train['Date'].dt.to_period('M')

# Group by 'YearMonth' and promotion combinations, then calculate average sales
monthly_avg_sales = attributes_train.groupby(['YearMonth', 'Promo', 'Promo2'])['Sales'].mean().reset_index()

# Convert 'YearMonth' back to datetime for continuous x-axis
monthly_avg_sales['YearMonth'] = monthly_avg_sales['YearMonth'].astype(str)
monthly_avg_sales['YearMonth'] = pd.to_datetime(monthly_avg_sales['YearMonth'])

# Create a column to label the four different promotion combinations
monthly_avg_sales['Promo_Combination'] = monthly_avg_sales.apply(
    lambda row: 'No Promo' if row['Promo'] == 0 and row['Promo2'] == 0 else
                'Promo' if row['Promo'] == 1 and row['Promo2'] == 0 else
                'Promo2' if row['Promo'] == 0 and row['Promo2'] == 1 else
                'Promo and Promo2', axis=1)

# Plot the line graph
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_sales, x='YearMonth', y='Sales', hue='Promo_Combination', marker='o')

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Average Monthly Sales')
plt.title('Monthly Average Sales for Different Promotion Combinations')
plt.legend(title='Promotion Type')
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()