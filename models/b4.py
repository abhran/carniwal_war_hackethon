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



train= pd.read_csv("miss/train.csv")
test= pd.read_csv("miss/test.csv")





testx=test[["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Demand","Discount_avail","charges_1","charges_2 (%)","Minimum_price","Maximum_price","charges_range","Product_Category_Grade"]]




tx=train[["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Demand","Discount_avail","charges_1","charges_2 (%)","Minimum_price","Maximum_price","charges_range","Product_Category_Grade"]]
ty=train["Selling_Price"].astype(float)



# x_train, x_test, y_train, y_test = train_test_split(tx, ty, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test=tx,tx,ty,ty

trainx=np.array(x_train)
valx=np.array(x_test)

# x_valid = df_valid[features].values
 # initialize xgboost model
# model = xgb.XGBRegressor(max_depth=6)
# model= xgb.XGBClassifier(
#  n_jobs=-1
#  )
# model=ensemble.GradientBoostingRegressor(n_estimators=255,max_depth=7,min_samples_split=2,learning_rate=0.1,loss='ls')
model=ensemble.GradientBoostingRegressor(n_estimators=885,max_depth=7,min_samples_split=2,learning_rate=0.026,loss='ls',random_state=48)
# model=ensemble.GradientBoostingRegressor(n_estimators=900,max_depth=7,min_samples_split=2,learning_rate=0.026,loss='ls',random_state=48)
# model=ensemble.GradientBoostingRegressor(n_estimators=1000,max_depth=7,min_samples_split=2,learning_rate=0.023,loss='ls')
# model=ensemble.GradientBoostingRegressor(n_estimators=9900,max_depth=7,min_samples_split=2,learning_rate=0.001,loss='ls')
# model=ensemble.GradientBoostingRegressor(n_estimators=370,max_depth=8,min_samples_split=2,learning_rate=0.035,loss='ls')


# model=ensemble.GradientBoostingRegressor(n_estimators=305,max_depth=5,min_samples_split=2,learning_rate=0.05,loss='ls')
# model=ensemble.GradientBoostingRegressor(n_estimators=290,max_depth=5,min_samples_split=2,learning_rate=0.04,loss='ls')
# model=sklearn.ensemble.AdaBoostRegressor(base_estimator=None, n_estimators=200, learning_rate=0.1, loss='square', random_state=2)
# model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
#  model=sklearn.linear_model.Ridge()
# model=sklearn.linear_model.Lasso()
# model=sklearn.linear_model.ElasticNet()
# model=sklearn.linear_model.LinearRegression()
# model=sklearn.linear_model.LogisticRegression()
# model=SVR(kernel='rbf')
# model=sklearn.linear_model.SGDRegressor(loss='squared_loss', alpha=0.001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.01, shuffle=True, verbose=2, epsilon=0.1, random_state=2, learning_rate='optimal', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)


# fit model on training data (ohe)
print("training your model")
model.fit(trainx, np.array(y_train))




print(model.score(trainx,np.array(y_train)))
print(model.score(valx,np.array(y_test)))



a=model.predict(trainx)
# print(a)
b=model.predict(valx)
# print(b)
c=model.predict(testx)
# print(c)

def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
print("training loss: ",root_mean_squared_error(np.array(y_train),a))
print("val loss: ",root_mean_squared_error(np.array(y_test),b))


# print(min(y_train))
# print(min(a))
# print(max(a))
# a=pd.DataFrame(a)

# a[a["Selling_Pri>0]
z=[]
for i in a:
    if i >0:
        z.append(i)
    else:
        z.append(1)

k=[]
for i in b:
    if i >0:
        k.append(i)
    else:
        k.append(1)

print("train error",mean_squared_log_error(np.array(y_train),z))
print("val error",mean_squared_log_error(np.array(y_test),k))
# print(max(0,100-mean_squared_log_error(np.array(y_train),b)))
# print(mean_squared_log_error(np.array(y_test),b))

d=[]
for i in c:
    if i >0:
        d.append(i)
    else:
        d.append(1)
# print(d)
d=pd.DataFrame(d)
# print(d.min)
# d=np.square(d)
dat=pd.read_csv("Book1.csv")
dat["Selling_Price"]=d
dat.to_csv("xgnew.csv",index=False,)