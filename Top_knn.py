import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# IMPORT AND DROP
# Restaurants
restaurants_df = pd.read_csv('./geoplaces2.csv')
restaurants_df = restaurants_df.set_index(['placeID']).drop(columns=['the_geom_meter','address','country','fax','zip','smoking_area','url','Rambience','franchise','area','other_services'])
res_accepted_pay = pd.read_csv('./chefmozaccepts.csv')
res_parking = pd.read_csv('./chefmozparking.csv')
# Users
user_df = pd.read_csv('./userprofile.csv')
user_df = user_df.set_index(['userID']).drop(columns=['smoker','ambience','hijos','marital_status','birth_year','interest','personality','religion','activity','color','weight','height'])
usr_payment = pd.read_csv('./userpayment.csv')
# Ratings
# {userID*, placeID*, rating} (* means key)
ratings_df = pd.read_csv('./rating_final.csv')
ratings_df = ratings_df.drop(columns=['food_rating', 'service_rating'])
ratings_df = ratings_df.set_index(['userID', 'placeID'])

# MAPPING AND ENCODING (OHE)

# User Payment
# {userID*, PAYMENT_cash  PAYMENT_credit_card  PAYMENT_debit_card}
map_user_payment = {
   'cash': 'cash',
   'bank_debit_cards': 'debit_card', 
   'MasterCard-Eurocard': 'credit_card', 
   'VISA': 'credit_card',
   'American_Express': 'credit_card',  
}
usr_payment['payment'] = usr_payment['Upayment'].apply(lambda x: map_user_payment[x])
usr_payment = usr_payment[['userID']].join(pd.get_dummies(usr_payment['payment']).add_prefix('PAYMENT_')).groupby('userID').max()

# Restaurants
map_restaurant_payment = {
   'cash': 'cash',
   'bank_debit_cards': 'debit_card', 
   'MasterCard-Eurocard': 'credit_card', 
   'VISA': 'credit_card',
   'Visa': 'credit_card',
   'American_Express': 'credit_card',  
   'Japan_Credit_Bureau': 'credit_card',  
   'Carte_Blanche': 'credit_card',
   'Diners_Club': 'credit_card', 
   'Discover': 'credit_card', 
   'gift_certificates': 'other',  
   'checks': 'other',  
}
res_accepted_pay['payment'] = res_accepted_pay['Rpayment'].apply(lambda x: map_restaurant_payment[x])
res_accepted_pay = res_accepted_pay[['placeID']].join(pd.get_dummies(res_accepted_pay['payment']).add_prefix('PAYMENT_')).groupby('placeID').max()

map_restaurant_parking = {
   'none': 'no',
   'public': 'yes', 
   'yes': 'yes', 
   'valet parking': 'yes',
   'fee': 'yes',
   'street': 'street',  
   'validated parking': 'yes',  
}
res_parking['parking'] = res_parking['parking_lot'].apply(lambda x: map_restaurant_parking[x])
res_parking = res_parking[['placeID']].join(pd.get_dummies(res_parking['parking']).add_prefix('PARKING_')).groupby('placeID').max()

# MERGE
# Users
# {userID*, 'latitude', 'longitude', 'drink_level', 'dress_preference', 'transport', 'budget', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card'}
user_df = pd.merge(user_df, usr_payment, how='left', on=['userID'])

# Map user feature values
user_df.drink_level = user_df.drink_level.map({'abstemious':1,'casual drinker':2,'social drinker':3})
user_df.dress_preference = user_df.dress_preference.map({'no preference':1,'informal':2,'formal':3,'elegant':4,'?':1})
user_df.transport = user_df.transport.map({'public':1,'on foot':2,'car owner':3,'?':1})
user_df.budget = user_df.budget.map({'low': 2, 'medium': 1, 'high': 3,'?':1})
user_df = user_df.fillna(0)

# Restaurants
# {placeID*,'latitude', 'longitude', 'name', 'city', 'state', 'alcohol', 'dress_code', 'accessibility', 'price', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes'}
restaurants_df = pd.merge(restaurants_df, res_accepted_pay, how='left', on=['placeID'])
restaurants_df = pd.merge(restaurants_df, res_parking, how='left', on=['placeID'])
# Map restaurant feature values
restaurants_df.alcohol = restaurants_df.alcohol.map({'No_Alcohol_Served':1,'Wine-Beer':2,'Full_Bar':3})
restaurants_df.dress_code = restaurants_df.dress_code.map({'informal':1,'casual':2,'formal':3})
restaurants_df.accessibility = restaurants_df.accessibility.map({'no_accessibility':1,'completely':2,'partially':3})
restaurants_df.price = restaurants_df.price.map({'low': 2, 'medium': 1, 'high': 3})
restaurants_df = restaurants_df.fillna(0)
# At this point, ratings_df, user_df and restaurants_df are clean

#-------------Simlar restaurant-------------
def get_index_from_restaurant(name):
    return restaurants_df[restaurants_df["name"]==name].index.tolist()[0]

def print_similar_restaurant(name):
    found_id = get_index_from_restaurant(name)
    for id in indices[found_id][1:]:
        print(restaurants_df.iloc[id]["name"])

def similar_restaurant(k):
    restaurants_df.reset_index(inplace = True)
    knn_res_df = restaurants_df.drop(['latitude', 'longitude', 'name', 'city', 'state'], axis=1)
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree').fit(knn_res_df)
    distances, indices = nbrs.kneighbors(knn_res_df)
    return indices

indices = similar_restaurant(6)
print('Top 5 restaurants:')
print_similar_restaurant("TACOS EL GUERO")

print('--------------------------------------')

#-------------Simlar user-------------
user_test = user_df.loc['U1001'].values

def print_similar_user(users):
    user_df_mod = user_df.reset_index()
    user_df_mod = user_df_mod.iloc[users]['userID'].drop(0, axis=0)
    return user_df_mod.values

def similar_user(user_test,k):
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree').fit(user_df)
    distances, indices = nbrs.kneighbors([user_test])
    return indices

indices = similar_user(user_test,6)
indices = np.squeeze(indices)
similar_users = print_similar_user(indices)
print('Top 5 users:')
print(similar_users)
