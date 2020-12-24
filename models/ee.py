import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Embedding,Concatenate
# models = []
import pandas as pd 
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
import numpy as np
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.preprocessing import PowerTransformer

from keras.callbacks import ModelCheckpoint
import keras.objectives
from keras.callbacks import CSVLogger




import pandas as pd
from sklearn import preprocessing


train= pd.read_csv("data/train.csv")
test= pd.read_csv("data/test.csv")

train=train[train["Selling_Price"]>0]
train["Selling_Price"]=train["Selling_Price"].fillna(train["Selling_Price"].mean())
test.loc[:, "Selling_Price"] = -1000000000
# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
data=data.drop(columns=['Customer_name','instock_date','Product_id'])






a=data["Minimum_price"].mean()
b=data["Maximum_price"].mean()
c=b-a





data["charges_1"]=data["charges_1"].fillna(data["charges_1"].mean())
data["charges_2 (%)"]=data["charges_2 (%)"].fillna(data["charges_2 (%)"].mean())


data["Minimum_price"]=data["Minimum_price"].fillna(data["Maximum_price"]-c)
data["Maximum_price"]=data["Maximum_price"].fillna(data["Minimum_price"]+c)

data["Minimum_price"]=data["Minimum_price"].fillna(data["Minimum_price"].mean())
data["Maximum_price"]=data["Maximum_price"].fillna(data["Maximum_price"].mean())



# pq=data[["Demand","charges_1","Minimum_price","Maximum_price","Selling_Price"]]
pq=data[["Demand","charges_1","Minimum_price","Maximum_price"]]
from sklearn import preprocessing
normalized_X = preprocessing.normalize(pq)
# data[["Demand","charges_1","Minimum_price","Maximum_price","Selling_Price"]]=normalized_X 
data[["Demand","charges_1","Minimum_price","Maximum_price"]]=normalized_X 



for feat in ["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","charges_2 (%)"]:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("Rare").astype(str).values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
train2= data[data.Selling_Price != -1000000000].reset_index(drop=True)
test2 = data[data.Selling_Price == -1000000000].reset_index(drop=True)








csv_logger = CSVLogger(f'swap/loss.csv', append=True, separator=',')

# train=pd.read_csv("trainee.csv")
# test=pd.read_csv("teste.csv")

# q=train[(train["UnitPrice"]>5000) ].index
# print(q)
# train=train.drop(q)
train= shuffle(train2, random_state=20)

# print(data.info())
# print(data.describe())
# print(train.info())
# print(train.describe())
# print(test.info())
# print(test.describe())

tx=train[["Stall_no","Market_Category","Loyalty_customer","Product_Category","Grade","Discount_avail","charges_2 (%)"]]

ty=train["Selling_Price"].astype(float)
ty=np.array(ty)




x_train=tx
y_train=ty
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
x6=x_train["Discount_avail"].astype(int)
x6=np.array(x6)
x7=x_train["charges_2 (%)"].astype(int)
x7=np.array(x7)

contidata=train[["Demand","charges_1","Minimum_price","Maximum_price"]]

inputs = []
embeddings = []

conti = Input(shape=(4,))
contiout=x = Dense(8, activation='relu')(conti)
inputs.append(conti)
embeddings.append(contiout)




invoice = Input(shape=(1,))
embedding = Embedding(51,10, input_length=1)(invoice)
embedding = Reshape(target_shape=(10,))(embedding)
inputs.append(invoice)
embeddings.append(embedding)
  
stockcode = Input(shape=(1,))
embedding = Embedding(275,25, input_length=1)(stockcode)
embedding = Reshape(target_shape=(25,))(embedding)
inputs.append(stockcode)
embeddings.append(embedding)

quantity = Input(shape=(1,))
embedding = Embedding(2,2, input_length=1)(quantity)
embedding = Reshape(target_shape=(2,))(embedding)
inputs.append(quantity)
embeddings.append(embedding)
    
customerid = Input(shape=(1,))
embedding = Embedding(10,4, input_length=1)(customerid)
embedding = Reshape(target_shape=(4,))(embedding)
inputs.append(customerid)
embeddings.append(embedding)

country = Input(shape=(1,))
embedding = Embedding(4,3, input_length=1)(country)
embedding = Reshape(target_shape=(3,))(embedding)
inputs.append(country)
embeddings.append(embedding)
  
  
year = Input(shape=(1,))
embedding = Embedding(3,2, input_length=1)(year)
embedding = Reshape(target_shape=(2,))(embedding)
inputs.append(year)
embeddings.append(embedding)
  
months = Input(shape=(1,))
embedding = Embedding(18,5, input_length=1)(months)
embedding = Reshape(target_shape=(5,))(embedding)
inputs.append(months)
embeddings.append(embedding)
  

x = Concatenate()(embeddings)
x = Dense(128, activation='relu')(x)
x = Dropout(.25)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(.25)(x)
# x=tf.keras.layers.BatchNormalization()(x)
# x=tf.expand_dims(x,axis=-1)
# x=tf.keras.layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2)(x)
x = Dense(64, activation='relu')(x)
# x=tf.keras.layers.BatchNormalization()(x)
x = Dropout(.15)(x)
x = Dense(64, activation='relu')(x)
# x=tf.keras.layers.BatchNormalization()(x)
x = Dropout(.15)(x)
# x=tf.expand_dims(x,axis=-1)
# x=tf.keras.layers.LSTM(32,dropout=0.2,recurrent_dropout=0.2)(x)
# x = Dropout(.15)(x)
x = Dense(8, activation='relu')(x)
# x = Dropout(.15)(x)
output = Dense(1)(x)
model = Model(inputs, output)
# model = Model(inputs, output)


from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))




def lr_scheduler(epoch, lr):
        if epoch%50==0:
                return lr * 0.4
        return lr

callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)





# opt=keras.optimizers.RMSprop(lr=1e-4,decay=0.95,momentum=0.9, epsilon=1e-8, name="RMSprop")
opt=keras.optimizers.Adam(lr=1e-3)
model.compile(loss = 'mean_absolute_error',optimizer=opt, metrics=[root_mean_squared_error])
# model.compile(loss =root_mean_squared_error ,optimizer="adam", metrics=['mean_absolute_error'])
print(model.summary())


cp1= ModelCheckpoint(filepath="swap/save_best.h5", monitor='loss',save_best_only=True, mode='min',verbose=1,save_weights_only=True)
cp2= ModelCheckpoint(filepath='swap/save_all.h5', monitor='loss',save_best_only=False ,verbose=1,save_weights_only=True)
# callbacks_list = [callback,cp1,cp2,csv_logger]
callbacks_list = [callback,cp1,cp2,csv_logger]



# X = shuffle([x1,x2,x3], random_state=20)
# ty = shuffle(ty, random_state=20)
# X_train, X_test, y_train, y_test = train_test_split([x1,x2,x3], ty, test_size=0.2, random_state=42)


# validation_data=([x11,x22,x33,x44,x55,x66,x77],y_test),
# model.fit(X_train,y_train, batch_size =1024, epochs = 1000, validation_split = 0.2,validation_data=(xval, yval))
model.fit([contidata,x1,x2,x3,x4,x5,x6,x7],y_train, batch_size =64, epochs =500,shuffle=True)#,callbacks=[callbacks_list])
model_json = model.to_json()
with open(f"swap/model.json", "w") as json_file:
    json_file.write(model_json)
# model.save_weights(f"modelswap.h5")