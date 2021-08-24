#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import jsonify
import json
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
import joblib
import flask
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
import os
from math import radians
from sklearn.metrics.pairwise import haversine_distances

import warnings
warnings.filterwarnings("ignore")

# In[ ]:


order_seller = joblib.load("order_seller_table.pkl")
total_order_id = joblib.load("total_order_id.pkl")
total_seller_id = joblib.load("total_seller_id.pkl")
user_order = joblib.load("user_order_table.pkl")
user_total = joblib.load("user_total.pkl")
order_total = joblib.load("order_total.pkl")
    
cat_seller = joblib.load("cat_seller_table.pkl")
total_cat_order_id = joblib.load("total_cat_order_id.pkl")
total_cat_seller_id = joblib.load("total_cat_seller_id.pkl")
user_cat = joblib.load("user_cat_table.pkl")
user_cat_total = joblib.load("user_cat_total.pkl")
order_cat_total = joblib.load("order_cat_total.pkl")

dict_seller = joblib.load('dict_seller.pkl')
dict_customer = joblib.load('dict_customer.pkl')
dict_seller_order = joblib.load("dict_seller_order.pkl")

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


# In[ ]:


import flask
app = Flask(__name__)


# In[ ]:


@app.route('/')
def hello_world():
    return 'Hello World!'


# In[ ]:


@app.route('/index/')
def index():
    return flask.render_template('index.html')

# In[ ]:

def model_predict(filepath):
    
    x = json.load(open(filepath))
    x = pd.Series(x)
   
    ##### Convert to datetime type ####
    x["order_purchase_timestamp"] = pd.to_datetime(x["order_purchase_timestamp"])
    x["order_approved_at"] = pd.to_datetime(x["order_approved_at"])
    x["order_delivered_carrier_date"] = pd.to_datetime(x["order_delivered_carrier_date"])
    x["order_delivered_customer_date"] = pd.to_datetime(x["order_delivered_customer_date"])
    x["order_estimated_delivery_date"] = pd.to_datetime(x["order_estimated_delivery_date"])
    x["shipping_limit_date"] = pd.to_datetime(x["shipping_limit_date"])

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
    cust_loc   = np.array([radians(float(x.lat_customer)),radians(float(x.lng_customer))])
    seller_loc = np.array([radians(float(x.lat_seller)),radians(float(x.lng_seller))])
    
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
    x["size"] = float(x["product_length_cm"])*float(x["product_height_cm"])*float(x["product_width_cm"])
    
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

    x["cust_category_share"] =user_cat.loc[(x["product_category_name"],x["customer_unique_id"])]/order_cat_total[x["product_category_name"]]
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
    x["order_item_id"] = str(x["order_item_id"])
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
        x["order_item_id"] = str(x["order_item_id"])
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
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return render_template("output.html",data=preds)
    return None

    
    
    
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8080,threaded=False)


# In[ ]:




