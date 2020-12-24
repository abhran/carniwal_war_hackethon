import pandas as pd 



train= pd.read_csv("data/train.csv")
test= pd.read_csv("data/test.csv")


test=test2[["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Demand","Discount_avail","charges_1","charges_2 (%)","Minimum_price","Maximum_price"]]