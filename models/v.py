import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
import sklearn
from sklearn.svm import SVR

train= pd.read_csv("data/train.csv")
test= pd.read_csv("data/test.csv")








# a=train["Product_id"].unique()
# b=test["Product_id"].unique()
# print(train.info())
# print(test.info())
# print(train.describe())
# print(test.describe())
# print(a.shape)
# print(b.shape)
# train=train.dropna()
# test=test.dropna()
# a=train["Product_id"].unique()
# b=test["Product_id"].unique()
# print(a.shape)
# print(b.shape)
train["Selling_Price"]=train["Selling_Price"].fillna(train["Selling_Price"].mean())
test.loc[:, "Selling_Price"] = -1000000000
# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
data=data.drop(columns=['Customer_name','instock_date','Product_id'])
# data=data.drop('instock_date')
# data=data.drop(' Product_id')

data["charges_1"]=data["charges_1"].fillna(data["charges_1"].mean())
data["charges_2 (%)"]=data["charges_2 (%)"].fillna(data["charges_2 (%)"].mean())
data["Minimum_price"]=data["Minimum_price"].fillna(data["Minimum_price"].mean())
data["Maximum_price"]=data["Maximum_price"].fillna(data["Maximum_price"].mean())

# "charges_1","charges_2 (%)"","Minimum_price","Maximum_price","Selling_Price"





for feat in ["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail"]:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("Rare").astype(str).values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
train2= data[data.Selling_Price != -1000000000].reset_index(drop=True)
test2 = data[data.Selling_Price == -1000000000].reset_index(drop=True)

# print(train2.info())
# print(test2.info())

# print(train2.describe())
# print(test2.describe())









tx=train2[["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Demand","Discount_avail","charges_1","charges_2 (%)","Minimum_price","Maximum_price"]]
ty=train2["Selling_Price"].astype(float)



x_train, x_test, y_train, y_test = train_test_split(tx, ty, test_size=0.2, random_state=42)


x1=x_train["Stall_no"].astype(int)
x1=np.array(x1)
x2=x_train["Market_Category"].astype(int)
x2=np.array(x2)
x3=x_train["Loyalty_customer"].astype(int)
x3=np.array(x3)
x4=x_train["Product_Category"].astype(int)
x4=np.array(x4)
x5=x_train["Grade"].astype(int)
x5=np.array(x5)
x6=x_train["Demand"].astype(int)
x6=np.array(x6)
x7=x_train["Discount_avail"].astype(int)
x7=np.array(x7)
x8=x_train["charges_1"].astype(int)
x8=np.array(x8)
x9=x_train["charges_2 (%)"].astype(int)
x9=np.array(x9)
x10=x_train["Minimum_price"].astype(int)
x10=np.array(x10)
x11=x_train["Maximum_price"].astype(int)
x11=np.array(x11)


x1s=x_test["Stall_no"].astype(int)
x1s=np.array(x1s)
x2s=x_test["Market_Category"].astype(int)
x2s=np.array(x2s)
x3s=x_test["Loyalty_customer"].astype(int)
x3s=np.array(x3s)
x4s=x_test["Product_Category"].astype(int)
x4s=np.array(x4s)
x5s=x_test["Grade"].astype(int)
x5s=np.array(x5s)
x6s=x_test["Demand"].astype(int)
x6s=np.array(x6s)
x7s=x_test["Discount_avail"].astype(int)
x7s=np.array(x7s)
x8s=x_test["charges_1"].astype(int)
x8s=np.array(x8s)
x9s=x_test["charges_2 (%)"].astype(int)
x9s=np.array(x9s)
x10s=x_test["Minimum_price"].astype(int)
x10s=np.array(x10s)
x11s=x_test["Maximum_price"].astype(int)
x11s=np.array(x11s)


trainx=np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
trainx=np.reshape(trainx,(5094,11))
valx=np.array([x1s,x2s,x3s,x4s,x5s,x6s,x7s,x8s,x9s,x10s,x11s])
valx=np.reshape(valx,(1274,11))
# test=np.array([x111,x222,x333,x444,x555,x666,x777])
# test=np.reshape(test,(122049,7))



# x_valid = df_valid[features].values
 # initialize xgboost model
# model = xgb.XGBRegressor(max_depth=1)
# model= xgb.XGBClassifier(
#  n_jobs=-1
#  )

# model=ensemble.GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split=2,learning_rate=0.1,loss='ls')
# model=sklearn.linear_model.Ridge()
# model=sklearn.linear_model.Lasso()
# model=sklearn.linear_model.ElasticNet()
# model=sklearn.linear_model.LinearRegression()
# model=sklearn.linear_model.LogisticRegression()
model=SVR()


 # fit model on training data (ohe)
print("training your model")
model.fit(trainx, np.array(y_train))




print(model.score(trainx,np.array(y_train)))
print(model.score(valx,np.array(y_test)))




a=model.predict(trainx)
# print(a)
b=model.predict(valx)
# print(b)
c=model.predict(test2.drop(columns=["Selling_Price"]))
# print(c)

def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
print(root_mean_squared_error(np.array(y_train),a))
print(root_mean_squared_error(np.array(y_test),b))
# c=pd.DataFrame(c)
# c.to_csv("xg2.csv")