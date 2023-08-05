import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

def mean_non_Nan(val):
  n = np.sum(~np.isnan(val))
  val.fillna(0, inplace = True)
  avg = val.sum()/n
  return avg

# Handling bookings.csv
df = pd.read_csv('/root_folder/data/bookings.csv')
epoch_date = "1997-01-01 00:00:00"
epoch_date = pd.to_datetime(epoch_date)
df['booking_status'] = pd.factorize(df['booking_status'])[0] + 1
df['booking_create_timestamp'] = pd.to_datetime(df['booking_create_timestamp']) - epoch_date
df['booking_approved_at'] = pd.to_datetime(df['booking_approved_at']) -epoch_date
df['booking_checkin_customer_date'] = pd.to_datetime(df['booking_checkin_customer_date']) - epoch_date
df['booking_time_taken'] = df['booking_approved_at'] - df['booking_create_timestamp']
df['booking_time_taken'] = df['booking_time_taken'].dt.total_seconds()
df['booking_time_crea_check'] = df['booking_checkin_customer_date'] - df['booking_create_timestamp']
df['booking_time_crea_check'] = df['booking_time_crea_check'].dt.total_seconds()

#Handling bookings_data.csv
df2 = pd.read_csv('/root_folder/data/bookings_data.csv')
df2['seller_agent_id'] = pd.factorize(df2['seller_agent_id'])[0] + 1
df2['booking_expiry_date'] = pd.to_datetime(df2['booking_expiry_date'])
epoch_updated = "1997-01-01 00:00:00"
epoch_updated = pd.to_datetime(epoch_updated)
df2['booking_expiry_date'] = df2['booking_expiry_date'] - epoch_updated

#Handling customer_data.csv
df3 = pd.read_csv('/root_folder/data/customer_data.csv')
df3['country'] = pd.factorize(df3['country'])[0] + 1
df3['customer_unique_id'] = pd.factorize(df3['customer_unique_id'])[0] + 1

#Reading hotels data
df4 = pd.read_csv('/root_folder/data/hotels_data.csv')

#Handling payments_data.csv
df5 = pd.read_csv('/root_folder/data/payments_data.csv')
df5['payment_type'] = pd.factorize(df5['payment_type'])[0] + 1
df5 = df5.groupby(['booking_id']).agg({'payment_installments' :'mean', 'payment_sequential' : 'max',  'payment_type' : 'mean'})

#Merging bookings_data.csv and hotels_data.csv based on hotel_id
df_24 = pd.merge(df2, df4, on = "hotel_id", how = "outer")
df_24 = df_24.groupby('booking_id', as_index = False).agg({
                         'booking_sequence_id' : 'max',
                         'price':'sum', 
                         'agent_fees':'sum', 
                         'hotel_category':'mean', 
                         'hotel_name_length' : 'mean',
                         'hotel_description_length' : 'mean',
                         'hotel_photos_qty' : 'mean',
                         'seller_agent_id' : 'mean',
                         'booking_expiry_date' : 'mean'
                         })

#Merging rest of the dataframes based on same booking_id
df_241 = pd.merge(df_24, df, on = "booking_id", how = "outer")
df_241['booking_time_exp_check'] = df_241['booking_expiry_date'] - df_241['booking_checkin_customer_date']
df_241['booking_time_exp_check'] = df_241['booking_time_exp_check'].dt.total_seconds()/1000
df_2415 = pd.merge(df_241, df5, on = "booking_id", how = "outer")
df_24153 = pd.merge(df_2415, df3, on = "customer_id", how = "outer")

df_24153['price'].fillna(mean_non_Nan(df_24153['price']))
df_24153['agent_fees'].fillna(mean_non_Nan(df_24153['agent_fees']))
df_24153['hotel_category'].fillna(mean_non_Nan(df_24153['hotel_category']))
df_24153['hotel_name_length'].fillna(mean_non_Nan(df_24153['hotel_name_length']))
df_24153['hotel_description_length'].fillna(mean_non_Nan(df_24153['hotel_description_length']))
df_24153['hotel_photos_qty'].fillna(mean_non_Nan(df_24153['hotel_photos_qty']))
df_24153['booking_time_taken'].fillna(mean_non_Nan(df_24153['booking_time_taken']))
df_24153['payment_sequential'].fillna(mean_non_Nan(df_24153['payment_sequential']))
df_24153['payment_installments'].fillna(mean_non_Nan(df_24153['payment_installments']))
df_24153['booking_sequence_id'].fillna(mean_non_Nan(df_24153['booking_sequence_id']))
df_24153['booking_time_crea_check'].fillna(mean_non_Nan(df_24153['booking_time_crea_check']))
df_24153['booking_time_exp_check'].fillna(mean_non_Nan(df_24153['booking_time_exp_check']))
df_24153['seller_agent_id'].fillna(mean_non_Nan(df_24153['seller_agent_id']))
df_24153['payment_type'].fillna(mean_non_Nan(df_24153['payment_type']))

#Get the training set columns
df_train = pd.read_csv('/root_folder/data/train_data.csv')
df_final_train = pd.merge(df_24153, df_train, on = "booking_id", how='inner')

#Get the relavant columns of training set and corresponding rating score given
final_train = df_final_train[['booking_sequence_id', 'price', 'agent_fees', 'hotel_category', 'hotel_name_length', 'hotel_description_length', 'hotel_photos_qty',  'booking_status', 'booking_time_taken', 'booking_time_crea_check', 'booking_time_exp_check', 'payment_installments', 'payment_sequential', 'payment_type', 'customer_unique_id', 'country']]
X = final_train.to_numpy()
y = df_final_train['rating_score'].values

#Get the test set columns data
df_test = pd.read_csv('/root_folder/data/sample_submission_5.csv')
df_final_test = pd.merge(df_24153, df_test, on = "booking_id", how = "right")
final_test = df_final_test[['booking_sequence_id', 'price', 'agent_fees', 'hotel_category', 'hotel_name_length', 'hotel_description_length',  'hotel_photos_qty','booking_status', 'booking_time_taken', 'booking_time_crea_check', 'booking_time_exp_check', 'payment_installments', 'payment_sequential', 'payment_type', 'customer_unique_id', 'country']]
XX = final_test.to_numpy()

#splitting dateset into train and validation set
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.33, stratify = y)

#Getting best hyperparameters
min = 1000
min_samples_leaf = [2, 3,5,7,9]
max_depth = [2, 3,5,7,9]
learning_rate = [0.01, 0.02, 0.05, 0.1, 0.2]
optimal_leaf = 2
optimal_lr = 0.01
optimal_depth = 2
for i in min_samples_leaf:
  for j in max_depth:
    for k in learning_rate:

        base_model = HistGradientBoostingRegressor(min_samples_leaf = i, max_depth = j, learning_rate = k).fit(X_train, y_train)

        y_train_pred = base_model.predict(X_train)
        y_valid_pred = base_model.predict(X_test)
        mse = mean_squared_error(y_test, y_valid_pred, squared = True)
        if(mse < min):
          min = mse
          optimal_lr = k
          optimal_depth = j
          optimal_leaf = i
        print("leaf : {},  depth : {} , learning_rate: {}, error : {}".format(i, j,k,  mean_squared_error(y_train, y_train_pred, squared = True)))
        print("leaf : {},  depth : {} ,  learning_rate: {},  error : {}".format(i, j,k, mean_squared_error(y_test, y_valid_pred, squared = True)))
        print()

print("optimal params : ",optimal_leaf, optimal_depth, optimal_lr)
print("least validation error : ", min)

#tuned_model = HistGradientBoostingRegressor(min_samples_leaf = 7, max_depth = 7, learning_rate = 0.05).fit(X_train, y_train)
tuned_model = HistGradientBoostingRegressor(min_samples_leaf = optimal_leaf, max_depth = optimal_depth, learning_rate = optimal_lr).fit(X, y)
y_test_pred_fin = tuned_model.predict(XX)

df___ = df_final_test.loc[:, ['booking_id', 'rating_score']]
df___['rating_score'] = y_test_pred_fin

df___.to_csv('/root_folder/output/CS20B022_CS20B051.csv', index = False)