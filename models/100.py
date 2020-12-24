
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PowerTransformer






import pandas as pd
from sklearn import preprocessing


train= pd.read_csv("trainm.csv")
test= pd.read_csv("data/test.csv")
test22= pd.read_csv("xg3.csv")
print(train.info())
print(test.info())
print(train.describe())
print(test.describe())
# print(train[train["Selling_Price"]>18000])
# print("hohohohohohohohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
# print(train[((train["Selling_Price"]<train["Minimum_price"]) & (train["Discount_avail"]==1.0)) |((train["Selling_Price"]>train["Minimum_price"]) & (train["Discount_avail"]==0.0))])
# print(train[(train["Selling_Price"]>train["Minimum_price"]) & (train["Discount_avail"]==0.0)])
# print("hhhhhhhhhhhhhhhhhhhhaaaaaaaaaaaaaaaaaaaaaaaaa")
# print(test22[test22["Selling_Price"]>18000])
# print(test.describe())
# train=train[train["Selling_Price"]>0]
# train["Selling_Price"]=train["Selling_Price"].fillna(train["Selling_Price"].mean())
# test.loc[:, "Selling_Price"] = -1000000000
# # concatenate both training and test data
# data = pd.concat([train, test]).reset_index(drop=True)
# data=data.drop(columns=['Customer_name','instock_date','Product_id'])






# a=data["Minimum_price"].mean()
# b=data["Maximum_price"].mean()
# c=b-a





# data["charges_1"]=data["charges_1"].fillna(data["charges_1"].mean())
# data["charges_2 (%)"]=data["charges_2 (%)"].fillna(data["charges_2 (%)"].mean())


# data["Minimum_price"]=data["Minimum_price"].fillna(data["Maximum_price"]-c)
# data["Maximum_price"]=data["Maximum_price"].fillna(data["Minimum_price"]+c)

# data["Minimum_price"]=data["Minimum_price"].fillna(data["Minimum_price"].mean())
# data["Maximum_price"]=data["Maximum_price"].fillna(data["Maximum_price"].mean())



# # pq=data[["Demand","charges_1","Minimum_price","Maximum_price","Selling_Price"]]
# pq=data[["Demand","charges_1","Minimum_price","Maximum_price"]]
# from sklearn import preprocessing
# normalized_X = preprocessing.normalize(pq)
# # data[["Demand","charges_1","Minimum_price","Maximum_price"]]=normalized_X 
# # data[["Demand","charges_1","Minimum_price","Maximum_price","Selling_Price"]]=normalized_X 





# for feat in ["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","charges_2 (%)"]:
#     lbl_enc = preprocessing.LabelEncoder()
#     temp_col = data[feat].fillna("Rare").astype(str).values
#     data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
# # print(data.info())
# # print(data.describe())
# r=[i for i in range(0,9743)]
# import matplotlib.pyplot as plt
# p=data["Demand"]
# q=data["charges_1"]








# plt.scatter(r,p)
# plt.show()
# plt.scatter(r,q)
# plt.show()
# train2= data[data.Selling_Price != -1000000000].reset_index(drop=True)
# test2 = data[data.Selling_Price == -1000000000].reset_index(drop=True)


