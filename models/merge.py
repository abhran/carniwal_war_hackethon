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

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")




def ide(x):
    return x[:4]
def invert(x):
    if x<=0:
        return x*(-1)
    return x
train=train[train["Selling_Price"]>0]
# train["Selling_Price"]=train["Selling_Price"].apply(invert)
train["Selling_Price"]=train["Selling_Price"].fillna(train["Selling_Price"].mean())
test.loc[:, "Selling_Price"] = -1000000000
# concatenate both training and test data

df= pd.concat([train, test]).reset_index(drop=True)
df["ide"]=df['Product_id'].apply(ide)


a=df["Minimum_price"].mean()
b=df["Maximum_price"].mean()
c=b-a

df["charges_1"]=df["charges_1"].fillna(df["charges_1"].mean())
df["charges_2 (%)"]=df["charges_2 (%)"].fillna(df["charges_2 (%)"].mean())
df["Minimum_price"]=df["Minimum_price"].fillna(df["Maximum_price"]-c)
df["Maximum_price"]=df["Maximum_price"].fillna(df["Minimum_price"]+c)

df["Minimum_price"]=df["Minimum_price"].fillna(df["Minimum_price"].mean())
df["Maximum_price"]=df["Maximum_price"].fillna(df["Maximum_price"].mean())




df['instock_date'] = pd.to_datetime(df['instock_date'], errors='coerce')
df.loc[:, 'year'] = df['instock_date'].dt.year
df.loc[:, 'weekofyear'] = df['instock_date'].dt.weekofyear
df.loc[:, 'month'] = df['instock_date'].dt.month
df.loc[:, 'dayofweek'] = df['instock_date'].dt.dayofweek
df.loc[:, 'hour'] = df['instock_date'].dt.hour
df=df.drop(columns=['Customer_name','instock_date','Product_id'])


# print(df.head(50))
k=[]
l=[]
g=["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","year","month"]
for i in ["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","year","month"]:
    
    for j in ["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","year","month"]:
        if f"{i}_{j}" not in g and i != j:
            df[f"{i}_{j}"] = ( df[i].astype(str)+ "_" + df[j].astype(str) )
            l.append(f"{i}_{j}")
        k.append(f"{i}_{j}")
        k.append(f"{j}_{i}")



m=["ide","Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","year","weekofyear","month","dayofweek","hour"]
m.extend(l)

for feat in m:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = df[feat].fillna("Rare").astype(str).values
    df.loc[:, feat] = lbl_enc.fit_transform(temp_col)
train2= df[df.Selling_Price != -1000000000].reset_index(drop=True)
test2 = df[df.Selling_Price == -1000000000].reset_index(drop=True)

# print(train2.info())
# print(train2.describe())

# print(test2.info())
# print(test2.describe())


print(train2.head(50))
print(test2.head(50))

train2.to_csv("data/train3.csv")
test2.to_csv("data/test3.csv")










