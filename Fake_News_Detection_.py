# # Fake and Real News Detection System

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re

import tensorflow

#for padding when the dataset is not long enough
from keras.preprocessing.sequence import pad_sequences

#for tokenization
from keras.preprocessing.text import Tokenizer  

#for splitting the data into test and train sets
from sklearn.model_selection import train_test_split

#for the model structure 
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D

from sklearn.metrics import classification_report, accuracy_score


fake_df = pd.read_csv('Fake.csv',error_bad_lines=False)
true_df = pd.read_csv('True.csv',error_bad_lines=False)


# ## Data cleaning

unknown_publishers = []
for index, row in enumerate( true_df.text.values):
  try:
    record = row.split('-', maxsplit= 1)
    record[1]
    assert(len(record[0])<120)
  except:
    unknown_publishers.append(index)


#dropping the row which dont have the contextual data,  that is at the index = 8970 
true_df = true_df.drop(8970, axis=0)

publisher = []
temp_text =[]

for index, row in enumerate(true_df.text.values):
  if index in unknown_publishers:
    temp_text.append(row)
    publisher.append('Unknown')
  else:
    record = row.split('-', maxsplit = 1)
    publisher.append(record[0].strip())
    temp_text.append(record[1].strip()) 


true_df['publisher'] = publisher
true_df['text']= temp_text



# storing the indices fo which the 'text' is empty in the fake_df
empty_fake_index = [index for index, text in enumerate( fake_df.text.tolist()) if str(text).strip()== ""]


true_df['text'] = true_df['title']+ " "+ true_df['text']
fake_df['text'] = fake_df['title'] + " " + fake_df['text']



true_df['text'] = true_df['text'].apply(lambda x: str(x).lower())
fake_df['text'] = fake_df['text'].apply(lambda x: str(x).lower())


# ## Preprocessing Text

true_df['class'] = 1
fake_df['class'] = 0

true_df = true_df[['text', 'class']]
fake_df = fake_df[['text', 'class']]

data= true_df.append(fake_df, ignore_index = True)

#function for removing the special character
def remove_special_char(x):
  x = re.sub(r'[^\w ]+', "", x)
  x = ' '.join(x.split())
  return x


data['text'] = data['text'].apply(lambda x: remove_special_char(x))


# ## Vectorization -- Word2Vec

get_ipython().system('pip install gensim')
import gensim

y = data['class'].values


#list comprehension method
X= [d.split() for d in data['text'].tolist()]


DIM = 100
w2v_model = gensim.models.Word2Vec(sentences= X, vector_size = DIM, window= 10, min_count=1)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)


X= tokenizer.texts_to_sequences(X)


plt.hist([len(x) for x in X], bins = 700)
plt.show()

nos = np.array([len(x) for x in X])
len( nos[nos>1000])

max_len = 1000
X= pad_sequences(X, maxlen = max_len)

vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index

def get_weight_matrix(model):
  weight_matrix = np.zeros((vocab_size, DIM))

  for word, i in vocab.items():
    weight_matrix[i]= model.wv[word]
  
  return weight_matrix

embedding_vector = get_weight_matrix(w2v_model)


# creating the model 

model= Sequential()
model.add(Embedding(vocab_size, output_dim = DIM, weights = [embedding_vector], input_length= max_len, trainable = False))
model.add(LSTM(units = 128))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])


model.summary()


X_train, X_test, y_train, y_test = train_test_split(X,y)


model.fit(X_train, y_train, validation_split= 0.3, epochs = 6 )

y_pred = (model.predict(X_test) >=0.5).astype(int)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))



