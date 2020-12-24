import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
import sklearn
from sklearn.svm import SVR



train= pd.read_csv("trainm.csv")
test= pd.read_csv("data/test.csv")
pr=pd.read_csv("xgnew.csv")

def ncat(x):
    if x<200:
        return 0
    if x>=200 and x<300:
        return 1
    if x>=300 and x<400:
        return 2
    if x>=400:
        return 3

def nnancat(x):
    if x<10000:
        return 0
    return 1

train=train[train["Selling_Price"]>0]
train=train[train["Minimum_price"]<=22000]
train=train[train["Maximum_price"]<=33000]
train.loc[:, "test_or_train"] = 1

train=train[((train["Selling_Price"]<train["Minimum_price"]) & (train["Discount_avail"]==1.0)) |((train["Selling_Price"]>train["Minimum_price"]) & (train["Discount_avail"]==0.0))]
# train=train[(((train["Selling_Price"]<train["Minimum_price"]) & (train["Discount_avail"]==1.0)) |((train["Selling_Price"]>train["Minimum_price"]) & (train["Discount_avail"]==0.0))) & (train["Selling_Price"]<train["Maximum_price"])]

test["Selling_Price"] = pr["Selling_Price"]
test.loc[:, "test_or_train"] = 0
# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
data=data.drop(columns=['Customer_name','instock_date','Product_id'])







a=data["Minimum_price"].mean()
b=data["Maximum_price"].mean()
c=b-a


data["charges_1cat"]=data["charges_1"].apply(nnancat)
data["charges_2 (%)cat"]=data["charges_2 (%)"].apply(nnancat)

data["charges_range"]=data["charges_1"].apply(ncat)
i,j="Product_Category","Grade"
data[f"{i}_{j}"] = ( data[i].astype(str)+ "_" + data[j].astype(str) )

for feat in ["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","Product_Category_Grade"]:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("Rare").astype(str).values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
data=data[["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Demand","Discount_avail","charges_1","charges_2 (%)","Minimum_price","Maximum_price","charges_range","Product_Category_Grade","Selling_Price","test_or_train"]]
import numpy as np
from sklearn import impute
knn_imputer = impute.KNNImputer(n_neighbors=2)
data=knn_imputer.fit_transform(data)
print(data)
print(data[0])
# print(data[:0])
print(data.shape)
data=pd.DataFrame(data,columns=["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Demand","Discount_avail","charges_1","charges_2 (%)","Minimum_price","Maximum_price","charges_range","Product_Category_Grade","Selling_Price","test_or_train"])
train2= data[data.test_or_train == 1].reset_index(drop=True)
test2 = data[data.test_or_train ==0].reset_index(drop=True)

train2.to_csv("best/train.csv")
test2.to_csv("best/test.csv")