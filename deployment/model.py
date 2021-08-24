#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import libraries....
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import datetime
import pickle
import joblib
from unidecode import unidecode
from math import radians
from sklearn.metrics.pairwise import haversine_distances

import warnings
warnings.filterwarnings("ignore")


# In[7]:


#load the data with all created features
data = pd.read_csv("data_with_advanced_features.csv")
data.drop("Unnamed: 0", inplace=True, axis=1)


# In[8]:


#label encoding of seller_id
label = LabelEncoder()
seller = label.fit_transform(data.seller_id)
data["seller_id"] = seller

#save the encoder
filename="seller_id_encode.pkl"
pickle.dump(label,open(filename,"wb"))

#label encoding of product id
label = LabelEncoder()
product = label.fit_transform(data.product_id)
data["product_id"] = product

# save the encoder
filename="product_id_encode.pkl"
pickle.dump(label,open(filename,"wb"))


# In[9]:


#creating class labels
binary = []
for i in range(len(data)):
    if data.review_score[i]==5:
        binary.append(1)
    else:
        binary.append(0)
        
data["binary_target"] = binary


# In[10]:


#target variable is review_score
Y = data["binary_target"]
X = data


# In[11]:


#train test split with test size 25% and 75% of data as train
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=10)


# In[13]:


#payment_type 
vec = CountVectorizer()
vec.fit(x_train["payment_type"].values)
x_tr_pay_type = vec.transform(x_train.payment_type.values)
#save as pickle file
filename = "count_vect_payment_1.pkl"
pickle.dump(vec,open(filename,"wb"))


# In[16]:


#order_item_id 
x_train.order_item_id = x_train.order_item_id.astype(str)
x_test.order_item_id = x_test.order_item_id.astype(str)
vec = CountVectorizer(vocabulary=range(1,22))
vec.fit(x_train["order_item_id"])
x_tr_id = vec.transform(x_train.order_item_id)
#save as pickle file
filename = "count_vect_item_1.pkl"
pickle.dump(vec,open(filename,"wb"))


# In[17]:


#product_category_name
vec = CountVectorizer()
vec.fit(x_train["product_category_name"].values)
x_tr_cat = vec.transform(x_train.product_category_name.values)
#save as pickle file
filename = "count_vect_cat_1.pkl"
pickle.dump(vec,open(filename,"wb"))


# In[18]:


x_tr_same_state = x_train.same_state.values.reshape(-1,1)
x_tr_same_city = x_train.same_city.values.reshape(-1,1)
x_tr_late_shipping = x_train.late_shipping.values.reshape(-1,1)
x_tr_high_freight = x_train.high_freight.values.reshape(-1,1)


# In[19]:


#data to be standardized
tr = x_train[["payment_sequential","payment_installments","payment_value","seller_id","product_id","seller_share","bu_share",
              "bs_share","cust_share",
          "lat_customer","lng_customer","lat_seller","lng_seller","product_name_lenght","product_description_lenght",
           "product_photos_qty","product_weight_g","size","price","delivery_day","delivery_date","delivery_month",
              "delivery_hour","purchased_day","purchased_date","purchased_month","purchased_hour","num_of_customers_for_seller",
              "num_of_sellers_for_cust","total_order_for_seller",
           "freight_value","estimated_time","actual_time","diff_actual_estimated","diff_purchased_approved",
           "diff_purchased_courrier","distance","speed","similarity","similarity_using_cat"]]


# In[20]:


norm = StandardScaler()
norm.fit(tr.values)
x_tr_num = norm.transform(tr.values)
#save as pickle file
filename = "std_num_1.pkl"
pickle.dump(norm,open(filename,"wb"))


# In[21]:


#horizontal stacking of all the features
train = hstack((x_tr_pay_type,x_tr_id,x_tr_cat,x_tr_num,x_tr_same_state,
                   x_tr_same_city,x_tr_late_shipping,x_tr_high_freight)).toarray()


# In[22]:


#reset the index of target variable
y_trains = y_train.reset_index()
y_train = y_trains["binary_target"]


# In[23]:


#logistic regression binary model
best_param = 0.01
model = LogisticRegression(C=best_param,class_weight="balanced")
model.fit(train,y_train)

#saving the logistic model as pickle file
filename = "binary_model.pkl"
pickle.dump(model,open(filename,"wb"))


# In[25]:


#custom ensemble for 1,2,3,4
#load the data with all created features
data = pd.read_csv("data_with_advanced_features.csv")
data.drop("Unnamed: 0", inplace=True, axis=1)

#label encoding of seller_id
label = LabelEncoder()
seller = label.fit_transform(data.seller_id)
data["seller_id"] = seller

filename = "seller_encode_2.pkl"
pickle.dump(label,open(filename,"wb"))

#label encoding of product id
label = LabelEncoder()
product = label.fit_transform(data.product_id)
data["product_id"] = product

filename = "product_encode_2.pkl"
pickle.dump(label,open(filename,"wb"))


# In[26]:


data = data[data["review_score"]!=5]
Y = data["review_score"]
X = data

######### train test split with test size 25% and 75% of data as train ##############
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=10)
###########################################################################################################

######## payment_type ##########
vec = CountVectorizer()
vec.fit(x_train["payment_type"].values)
x_tr_pay_type = vec.transform(x_train.payment_type.values)
# save as pickle file
filename = "countvec_pay_2.pkl"
pickle.dump(vec,open(filename,"wb"))

###### order_item_id ###########
x_train.order_item_id = x_train.order_item_id.astype(str)
vec = CountVectorizer(vocabulary=range(1,22))
vec.fit(x_train["order_item_id"])
x_tr_id = vec.transform(x_train.order_item_id)

# save as pickle file
filename = "countvec_item_2.pkl"
pickle.dump(vec,open(filename,"wb"))

######### product_category_name ############
vec = CountVectorizer()
vec.fit(x_train["product_category_name"].values)
x_tr_cat = vec.transform(x_train.product_category_name.values)
# save as pickle file
filename = "countvec_cat_2.pkl"
pickle.dump(vec,open(filename,"wb"))

########## Binary features #####################
x_tr_same_state = x_train.same_state.values.reshape(-1,1)
x_tr_same_city = x_train.same_city.values.reshape(-1,1)
x_tr_late_shipping = x_train.late_shipping.values.reshape(-1,1)
x_tr_high_freight = x_train.high_freight.values.reshape(-1,1)

################################################################################
############# data to be standardized #########################################
tr = x_train[["payment_sequential","payment_installments","payment_value","seller_id","product_id","seller_share","bu_share",
              "bs_share","cust_share",
          "lat_customer","lng_customer","lat_seller","lng_seller","product_name_lenght","product_description_lenght",
           "product_photos_qty","product_weight_g","size","price","delivery_day","delivery_date","delivery_month",
              "delivery_hour","purchased_day","purchased_date","purchased_month","purchased_hour","num_of_customers_for_seller",
              "num_of_sellers_for_cust","total_order_for_seller",
           "freight_value","estimated_time","actual_time","diff_actual_estimated","diff_purchased_approved",
           "diff_purchased_courrier","distance","speed","similarity","similarity_using_cat"]]


norm = StandardScaler()
norm.fit(tr.values)
x_tr_num = norm.transform(tr.values)
# save as pickle file
filename = "std_num_2.pkl"
pickle.dump(norm,open(filename,"wb"))
#################################################################################################################

#horizontal stacking of all the features
train = hstack((x_tr_pay_type,x_tr_id,x_tr_cat,x_tr_num,x_tr_same_state,
                   x_tr_same_city,x_tr_late_shipping,x_tr_high_freight)).toarray()


#reset the index of target variable
y_trains = y_train.reset_index()
y_train = y_trains["review_score"]


# In[27]:


def custom_ensemble(x_tr,y_tr,x_te,n_estimators,estimator,meta_clf):
    """This function creates the custom ensemble model and returns predicted  target variable of test set"""
    
    ########### SPlitting train data into 50-50 as d1 and d2 ############
    kf = StratifiedKFold(n_splits=2)
    
    d1 = x_tr[list(kf.split(x_tr,y_tr))[1][0]]
    d1_y = y_tr[list(kf.split(x_tr,y_tr))[1][0]]

    d2 = x_tr[list(kf.split(x_tr,y_tr))[1][1]]
    d2_y = y_tr[list(kf.split(x_tr,y_tr))[1][1]]
    #####################################################################
    d1_y = np.array(d1_y)
    d2_y = np.array(d2_y)
    #####################################################################
    ### Creating base learners and training them using samples of d1 ####
    
    models=[]
    
    for i in range(n_estimators):
        ind = np.random.choice(19387,size=(20000),replace=True)
        sample = d1[ind]
        sample_y = d1_y[ind]  
        
        estimator.fit(sample,sample_y)
        models.append(estimator)
    
    # save as pickle file
    filename="base_models.pkl"
    pickle.dump(models,open(filename,"wb"))
    ########### Predictions from base learners for d2 set ###############
    predictions = []
    for model in models: 
        
        pred = model.predict(d2)
        predictions.append(pred)
        
    predictions = np.array(predictions).reshape(-1,n_estimators)
    
    ########## meta classifier on predictions of base learners ##########
    
    meta_clf.fit(predictions,d2_y)
    
    # save as pickle file
    filename="meta_clf.pkl"
    pickle.dump(meta_clf,open(filename,"wb"))
    #####################################################################
    


# In[28]:


#training and saving the models with best hyperparameter n_estimator=150
best_n = 150
custom_ensemble(train,y_train,test,best_n,LogisticRegression(class_weight="balanced"),
                                       LogisticRegression(class_weight="balanced"))


# In[ ]:





# In[ ]:


order_seller = pd.read_pickle("order_seller_table.pkl")
total_order_id = pd.read_pickle("total_order_id.pkl")
total_seller_id = pd.read_pickle("total_seller_id.pkl")
user_order = pd.read_pickle("user_order_table.pkl")
user_total = pd.read_pickle("user_total.pkl")
order_total = pd.read_pickle("order_total.pkl")
    
cat_seller = pd.read_pickle("cat_seller_table.pkl")
total_cat_order_id = pd.read_pickle("total_cat_order_id.pkl")
total_cat_seller_id = pd.read_pickle("total_cat_seller_id.pkl")
user_cat = pd.read_pickle("user_cat_table.pkl")
user_cat_total = pd.read_pickle("user_cat_total.pkl")
order_cat_total = pd.read_pickle("order_cat_total.pkl")

dict_seller = pd.read_pickle('dict_seller.pkl')
dict_customer = pd.read_pickle('dict_customer.pkl')
dict_seller_order = pd.read_pickle("dict_seller_order.pkl")

label_seller = joblib.load("seller_id_encode.pkl")
label_prod = joblib.load("product_id_encode.pkl")

vec_pay = joblib.load("count_vect_payment_1.pkl")
vec_item = joblib.load("count_vect_item_1.pkl")
vec_cat = joblib.load("count_vect_cat_1.pkl")
norm_1 = joblib.load("std_num_1.pkl")

binary_model = joblib.load("binary_model.pkl")

label_seller_2 = joblib.load("seller_encode_2.pkl")
label_prod_2 = joblib.load("product_encode_2.pkl")
vec_pay_2 = joblib.load("countvec_pay_2.pkl")
vec_item_2 = joblib.load("countvec_item_2.pkl")
vec_cat_2 = joblib.load("countvec_cat_2.pkl")
norm_2 = joblib.load("std_num_2.pkl")

models = joblib.load("base_models.pkl")
meta_clf = joblib.load("meta_clf.pkl")

def predict(x):
    
    """This function takes single query point as input 
        and 
       preprocess,featurize itand finally gives the predicted target score"""
    
    ##### Convert to datetime type ####
    x["order_purchase_timestamp"] = pd.to_datetime(x["order_purchase_timestamp"])
    x["order_approved_at"] = pd.to_datetime(x["order_approved_at"])
    x["order_delivered_carrier_date"] = pd.to_datetime(x["order_delivered_carrier_date"])
    x["order_delivered_customer_date"] = pd.to_datetime(x["order_delivered_customer_date"])
    x["order_estimated_delivery_date"] = pd.to_datetime(x["order_estimated_delivery_date"])
    x["shipping_limit_date"] = pd.to_datetime(x["shipping_limit_date"])

    
    ################################ BASIC FEATURES ########################################
    #### Time based features #####
    #Time of estimated delivery
    x["estimated_time"] = round((x["order_estimated_delivery_date"]-x["order_purchase_timestamp"]).total_seconds()/3600,6)
    #Time taken for delivery
    x["actual_time"]    = round((x["order_delivered_customer_date"]-x["order_purchase_timestamp"]).total_seconds()/3600,6)
    #Difference between actual delivery time and estimated delivery time
    x["diff_actual_estimated"]   = round((x["order_delivered_customer_date"]-x["order_estimated_delivery_date"]).total_seconds()/3600,6)
    # difference between purchase time and approved time
    x["diff_purchased_approved"] = round((x["order_approved_at"]-x["order_purchase_timestamp"]).total_seconds()/3600,6)
    # difference between purchase time and courrier delivery time
    x["diff_purchased_courrier"] = round((x["order_delivered_carrier_date"]-x["order_purchase_timestamp"]).total_seconds()/3600,6)
    
    # some more features from timestamp(days, weekday, month, hour)
    x["delivery_day"]   = x["order_delivered_customer_date"].weekday()
    x["delivery_date"]  = x["order_delivered_customer_date"].day
    x["delivery_month"] = x["order_delivered_customer_date"].month
    x["delivery_hour"]  = x["order_delivered_customer_date"].hour

    x["purchased_day"]   = x["order_purchase_timestamp"].weekday()
    x["purchased_date"]  = x["order_purchase_timestamp"].day
    x["purchased_month"] = x["order_purchase_timestamp"].month
    x["purchased_hour"]  = x["order_purchase_timestamp"].hour
    
    
    
    ######## Distance based features ###############
    ### Distance between customer and seller ####
    cust_loc   = np.array([radians(x.lat_customer),radians(x.lng_customer)])
    seller_loc = np.array([radians(x.lat_seller),radians(x.lng_seller)])
    
    dist = haversine_distances([cust_loc, seller_loc])*6371
    x["distance"] = dist[0,1]
    
    ### Speed 
    x["speed"] = x["distance"]/x["actual_time"]
    
    ### Binary features like same city or not, same state or not ###
    ### same state
    x["same_state"] = 1 if (x.customer_state == x.seller_state) else 0
    
    ### same city 
    x["customer_city"] = unidecode(x["customer_city"].lower())
    x["seller_city"] = unidecode(x["seller_city"].lower())
    
    x["same_city"]   = 1 if (x.customer_city == x.seller_city) else 0
    
    ### late_shipping
    
    x["late_shipping"] = 1 if (x.shipping_limit_date < x.order_delivered_carrier_date) else 0
    
    ### high_freight
    x["high_freight"]  = 1 if (x.price < x.freight_value) else 0
    
    ### size of the product
    x["size"] = x["product_length_cm"]*x["product_height_cm"]*x["product_width_cm"]
    
    ########################## ADVANCED FEATURES ######################################
    ################# customer_seller similarity based on order_item_id ###############
    
    x["seller_share"] = order_seller.loc[(x["order_item_id"],x["seller_id"])]/total_order_id[x["order_item_id"]]
    x["bs_share"]     = order_seller.loc[(x["order_item_id"],x["seller_id"])]/total_seller_id[x["seller_id"]]
    
    x["cust_share"]   = user_order.loc[(x["order_item_id"],x["customer_unique_id"])]/order_total[x["order_item_id"]]
    x["bu_share"]     = user_order.loc[(x["order_item_id"],x["customer_unique_id"])]/user_total[x["customer_unique_id"]]
    
    ### similarity
    x["similarity"]   = np.dot([x["seller_share"],x["bs_share"]] , [x["cust_share"],x["bu_share"]])
    
    ######################## customer_seller similarity based on category_name ###############
    
    x["seller_category_share"] = cat_seller.loc[(x["product_category_name"],x["seller_id"])]/total_cat_order_id[x["product_category_name"]]
    x["cat_seller_share"]      = cat_seller.loc[(x["product_category_name"],x["seller_id"])]/total_cat_seller_id[x["seller_id"]]

    x["cust_category_share"]   = user_cat.loc[(x["product_category_name"],x["customer_unique_id"])]/order_cat_total[x["product_category_name"]]
    x["cat_cust_share"]        = user_cat.loc[(x["product_category_name"],x["customer_unique_id"])]/user_cat_total[x["customer_unique_id"]]

    ### similarity
    x["similarity_using_cat"]  = np.dot([x["seller_category_share"],x["cat_seller_share"]],[x["cust_category_share"],x["cat_cust_share"]])
    ############################################################################################
    ########### Total customers for each seller and total seller for each customer #############
 
    
    x["num_of_customers_for_seller"] = dict_seller[x["seller_id"]]
    x["num_of_sellers_for_cust"]     = dict_customer[x["customer_unique_id"]]
    x["total_order_for_seller"]      = dict_seller_order[x["seller_id"]]
    
#################################################################################################
    
    x["seller_id_label"] = label_seller.transform([x["seller_id"]])
    x["product_id_label"] = label_prod.transform([x["product_id"]])
    #################################################
    ############## countvectorizers ################
    ### payment_type
    x_te_pay_type = vec_pay.transform([x["payment_type"]])
    #### order_item_id
    x["order_item_id"] = x["order_item_id"].astype(str)
    x_te_id = vec_item.transform([x["order_item_id"]])
    ### product_category_name
    
    x_te_cat = vec_cat.transform([x["product_category_name"]])
    ####################################################
    ############### standardization ####################
    num = x[["payment_sequential","payment_installments","payment_value","seller_id_label","product_id_label","seller_share","bu_share",
              "bs_share","cust_share",
          "lat_customer","lng_customer","lat_seller","lng_seller","product_name_lenght","product_description_lenght",
           "product_photos_qty","product_weight_g","size","price","delivery_day","delivery_date","delivery_month",
              "delivery_hour","purchased_day","purchased_date","purchased_month","purchased_hour","num_of_customers_for_seller",
              "num_of_sellers_for_cust","total_order_for_seller",
           "freight_value","estimated_time","actual_time","diff_actual_estimated","diff_purchased_approved",
           "diff_purchased_courrier","distance","speed","similarity","similarity_using_cat"]]
    
    
    num = np.array(num).reshape(1,-1)
    x_te_num = norm_1.transform(num)
    #################################################################
    ######## concatenate all features to create query point #########
    query_point = hstack((x_te_pay_type,x_te_id,x_te_cat,x_te_num,x.same_state,
                              x.same_city,x.late_shipping,x.high_freight)).toarray()
    
    ##################################################################
    
    if binary_model.predict(query_point) == 1:
        prediction = 5
        
    else:
        ################ CUSTOM ENSEMBLE ##########################
        
        x["seller_id_enc"] = label_seller_2.transform([x["seller_id"]])
        x["product_id_enc"] = label_prod_2.transform([x["product_id"]])
        ####
        ######## countvectorizers ###########
        ### payment_type
        x_te_pay_type = vec_pay_2.transform([x["payment_type"]])
        
        #### order_item_id
        x["order_item_id"] = x["order_item_id"].astype(str)
        x_te_id = vec_item_2.transform([x["order_item_id"]])
        
        ### product_category_name
        x_te_cat = vec_cat_2.transform([x["product_category_name"]])
        
        ####################################################
        ############### standardization ####################
        num = x[["payment_sequential","payment_installments","payment_value","seller_id_enc","product_id_enc","seller_share","bu_share",
                 "bs_share","cust_share",
                 "lat_customer","lng_customer","lat_seller","lng_seller","product_name_lenght","product_description_lenght",
                 "product_photos_qty","product_weight_g","size","price","delivery_day","delivery_date","delivery_month",
                 "delivery_hour","purchased_day","purchased_date","purchased_month","purchased_hour","num_of_customers_for_seller",
                 "num_of_sellers_for_cust","total_order_for_seller",
                 "freight_value","estimated_time","actual_time","diff_actual_estimated","diff_purchased_approved",
                 "diff_purchased_courrier","distance","speed","similarity","similarity_using_cat"]]
    
        
        num = np.array(num).reshape(1,-1)
        x_te_num = norm_2.transform(num)
        #################################################################
        ######## concatenate all features to create query point #########
        query_point = hstack((x_te_pay_type,x_te_id,x_te_cat,x_te_num,x.same_state,
                              x.same_city,x.late_shipping,x.high_freight)).toarray()
        
        
        predicts = []
        for model in models:
            predicts.append(model.predict(query_point))
        predicts = np.array(predicts).reshape(1,-1)
        
        prediction = meta_clf.predict(predicts)
        
###############################################################################################################        
        
    return prediction     


# In[ ]:





# In[ ]:





# In[ ]:




