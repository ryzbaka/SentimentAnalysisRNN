import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM, Embedding,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
data=pd.read_csv('sent/train.csv')
data.drop(labels=273514,inplace=True)
def process_labels(x):
    return int(x)

data.label=data.label.map(lambda x: process_labels(x))

data.text=data.text.apply(lambda x: x.lower())
#converint to lowercase
data.text=data.text.apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
#removing all characters except alphanumeric ones

tokenizer=Tokenizer(num_words=5000,split=" ")
tokenizer.fit_on_texts(data['text'].values)

X=tokenizer.texts_to_sequences(data['text'].values)
X=pad_sequences(X)

y=pd.get_dummies(data['label']).values

model=Sequential()
model.add(Embedding(5000,256,input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))
model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


batch_size=32
epochs=8
model.fit(X,y,epochs=epochs,batch_size=batch_size,verbose=2)
