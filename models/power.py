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
from sklearn.preprocessing import PowerTransformer



train= pd.read_csv("data/train3.csv")
test= pd.read_csv("data/test3.csv")
 

testx=test.drop(columns=["Selling_Price"])
testx=testx.values

ty=train["Selling_Price"].astype(float)
tx=train.drop(columns=["Selling_Price"])

power = PowerTransformer(method='yeo-johnson', standardize=True)

ty=np.array(ty)
ty=ty.reshape(-1, 1)
# ty=np.reshape(ty,(6313,))

ty = power.fit_transform(ty)
print(np.max(ty))
print(np.min(ty))

ty=np.reshape(ty,(6313,))




# x_train, x_test, y_train, y_test = train_test_split(tx, ty, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test=tx.values,tx.values,ty,ty





trainx=np.array(x_train)
valx=np.array(x_test)


# x_valid = df_valid[features].values
 # initialize xgboost model
# model = xgb.XGBRegressor(max_depth=6)
# model= xgb.XGBClassifier(
#  n_jobs=-1
#  )
model=ensemble.GradientBoostingRegressor(n_estimators=455,max_depth=5,min_samples_split=2,learning_rate=0.05,loss='ls')
# model=ensemble.GradientBoostingRegressor(n_estimators=85,max_depth=8,min_samples_split=2,learning_rate=0.1,loss='ls')
# model=sklearn.ensemble.AdaBoostRegressor(base_estimator=None, n_estimators=400, learning_rate=0.1, loss='square', random_state=2)
# model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=9, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# model=sklearn.linear_model.Ridge()
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
a=np.array(a)
a=a.reshape(-1,1)
a = power.inverse_transform(a)

print(a)



b=model.predict(valx)
b=np.array(b)
b=b.reshape(-1,1)
b = power.inverse_transform(b)
print(b)


c=model.predict(testx)
c=np.array(c)
c=c.reshape(-1,1)
c = power.inverse_transform(c)
print(c)


y_train=np.array(y_train)
y_train=y_train.reshape(-1,1)
y_train = power.inverse_transform(y_train)

y_test=np.array(y_test)
y_test=y_test.reshape(-1,1)
y_test = power.inverse_transform(y_test)

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
dat=pd.read_csv("Book1.csv")
dat["Selling_Price"]=d
dat.to_csv("xg6.csv",index=False,)