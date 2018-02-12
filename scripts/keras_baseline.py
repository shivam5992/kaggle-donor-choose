from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional, SpatialDropout1D
from keras.layers.core import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence, text
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn import preprocessing
from nltk import word_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.models import Model
from keras import optimizers

from keras.layers import Merge, PReLU
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sparse
from collections import Counter 
from sklearn.preprocessing import LabelEncoder

import warnings
import time
import gc
import os
import string

punctuation = string.punctuation
warnings.filterwarnings('ignore')
start = time.time()

prepare_data = True 
feature_engg_flag = True
tf_idf_flag = False
model_flag = False
cleaning = False 

####################### Read Data ###############################################
if prepare_data:
  print ("reading data")

  id_column = "id"
  missing_token = " UNK "
  target_column = "project_is_approved"

  train = pd.read_csv("data/train.csv", index_col=id_column, parse_dates=["project_submitted_datetime"])
  test = pd.read_csv("data/test.csv", index_col=id_column, parse_dates=["project_submitted_datetime"])
  rc = pd.read_csv("data/resources.csv", index_col=id_column).fillna(missing_token)
  rc['total_price'] = rc.quantity * rc.price
  rc['price_sum'] = rc['price'].copy()
  rc['quantity_sum'] = rc['quantity'].copy()

  ###################### Datasets Preparation #####################################

  print ("dataset preparation")
  Y = train[target_column].copy()

  train = train.drop(target_column, axis=1) # axis:1 drop in all rows (vertical drop)
  df = pd.concat([train, test], axis=0) # axis:0 concatenate all columns together

  # aggregate resources data frame 
  agg_rc = rc.reset_index().groupby(id_column).agg(dict(price_sum='sum', quantity_sum='sum', total_price='mean', quantity='mean', price='mean', description=lambda x: missing_token.join(x)))

  # merge resources data frame with input dataframe
  df = pd.merge(df, agg_rc, left_index=True, right_index=True, how= "inner")

  traindex = train.index
  tesdex = test.index
  alldex = df.index
  del test, train
  del rc, agg_rc

#################### Feature Engineering ########################################
print ("feature engineering")

def cleanup_text(x):
  x = x.replace("\\r", " ").replace("\\t", " ").replace("\\n", " ")
  x = "".join(_ for _ in x if _ not in punctuation)
  return x.lower() 

def add_count_cat(tx, mapping):
    return sum([mapping[_.strip()] for _ in tx.split(",")]) / len(tx.split(","))

def getCountVar(compute_df, var_name, splitter=False):
  if splitter:
    values = []
    for each in df[var_name]:
      allval = each.split(",")
      allval = [x.strip() for x in allval]
      values.extend(allval)
    value_counts = dict(Counter(values))  
    compute_df["Count_"+var_name] = compute_df[var_name].apply(lambda x: add_count_cat(x, value_counts))    
  else:
    grouped_df = compute_df.groupby(var_name, as_index=False).agg('size').reset_index()
    grouped_df.columns = [var_name, "var_count"]
    merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
    merged_df.fillna(-1, inplace=True)
    compute_df["Count_"+var_name] = list(merged_df["var_count"])

drop_columns = ['teacher_id', 'project_essay_1','project_essay_2','project_essay_3','project_essay_4' ,'project_subject_categories',"project_subject_subcategories", "project_resource_summary","project_title","description","project_submitted_datetime"]
if feature_engg_flag:
  print ("generating new date fields")
  df["Year"] = df["project_submitted_datetime"].dt.year
  df["Date of Year"] = df['project_submitted_datetime'].dt.dayofyear
  df["Weekday"] = df['project_submitted_datetime'].dt.weekday
  df["Day of Month"] = df['project_submitted_datetime'].dt.day
  df["Month"] = df['project_submitted_datetime'].dt.month
  df["Hour"] = df['project_submitted_datetime'].dt.hour

  df["essay1_len"] = df['project_essay_1'].apply(len)
  df["essay2_len"] = df['project_essay_2'].apply(len)
  df["title_len"] = df['project_title'].apply(len)

  # create count columns
  cat_cols = ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'teacher_id']
  for col in cat_cols:
    if "subject" in col:
      getCountVar(df, col, splitter=True)
    else:
      getCountVar(df, col)   

  # label encoding
  catcols = ['project_subject_categories', 'project_subject_subcategories', 'Month', 'Hour' ,'Weekday','Day of Month','Year','Date of Year','teacher_prefix','school_state','project_grade_category']
  for c in catcols:
    le = LabelEncoder()
    le.fit(df[c].astype(str))
    df[c] = le.transform(df[c].astype(str))

  keep_columns = list(set(list(df.columns)) - set(drop_columns))
  std = StandardScaler()
  df[keep_columns] = pd.DataFrame(std.fit_transform(df[keep_columns])).set_index(alldex)

  # dont uncomment
  # df.drop(drop_columns, axis=1, inplace=True)
  # df.to_csv("models/preprocessed/normdf.csv")

  del std

if tf_idf_flag:
  print ("generating new text fields")
  df['project_title'] = df['project_title'].apply(lambda x : cleanup_text(x))
  df['project_essay_1'] = df['project_essay_1'].apply(lambda x : cleanup_text(x))
  df['project_essay_2'] = df['project_essay_2'].apply(lambda x : cleanup_text(x))
  
  df['project_essay_3'] = df['project_essay_3'].fillna(missing_token)
  df['project_essay_3'] = df['project_essay_3'].apply(lambda x : cleanup_text(x))
  
  df['project_essay_4'] = df['project_essay_4'].fillna(missing_token)
  df['project_essay_4'] = df['project_essay_4'].apply(lambda x : cleanup_text(x))
  
  df['project_resource_summary'] = df['project_resource_summary'].apply(lambda x : cleanup_text(x))

  df['description'] = df['description'].apply(lambda x : cleanup_text(x))

  df['text'] = df.apply(lambda row: ' '.join([str(row['project_essay_1']), str(row['project_essay_2']), 
                      str(row['project_essay_3']), str(row['project_essay_4']),
                      str(row['project_resource_summary']), str(row['project_title']),
                      str(row['description'])]), axis=1)
  df.drop(drop_columns, axis=1, inplace=True)
  
  # df.to_csv('models/preprocessed/processed_input.csv')

################### for combined model ###############################

train = df.loc[traindex,]
test = df.loc[tesdex,]
# del df 

tfeats = list(df.columns)
tfeats.remove('text')
train_feats = train[tfeats]
test_feats = test[tfeats]

train_feats.to_csv('models/preprocessed/train_feats.csv')
test_feats.to_csv('models/preprocessed/test_feats.csv')
# exit(0)


########################### start modelling ###############################

print ("splitting valid")
# xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, Y, random_state=42, test_size=0.1, shuffle=True)
xtrain = train.project_title.values
xtest = test.project_title.values

print ("load the fast text vectors in a dictionary")
embeddings_index = {}
count = 0

# f = open('data/wiki-news-300d-1M.vec') # for python 2
f = open('data/wiki-news-300d-1M.vec', encoding="utf8")
for line in tqdm(f):
    count += 1 
    if count == 10:
      break
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# def sent2vec(s):
#     words = str(s).lower()
#     words = word_tokenize(words)
#     words = [w for w in words if w.isalpha()]
#     M = []
#     for w in words:
#         try:
#             M.append(embeddings_index[w])
#         except:
#             continue
#     M = np.array(M)
#     v = M.sum(axis=0)
#     if type(v) != np.ndarray:
#         return np.zeros(300)
#     return v / np.sqrt((v ** 2).sum())

# print ("create sentence vectors using the above function for training and validation set")
# xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]
# xtest_glove = [sent2vec(x) for x in tqdm(xtest)]

# xtrain_glove = np.array(xtrain_glove)
# xtest_glove = np.array(xtest_glove)

# print("scale the data before any neural net")
# scl = preprocessing.StandardScaler()
# xtrain_glove_scl = scl.fit_transform(xtrain_glove)
# xtest_glove_scl = scl.transform(xtest_glove)


print ("keras preprocessing")
token = text.Tokenizer(num_words=100000)
max_len = 300

token.fit_on_texts(list(xtrain))
xtrain_seq = token.texts_to_sequences(xtrain)
xtest_seq = token.texts_to_sequences(xtest)

print ("zero pad the sequences")
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

word_index = token.word_index

print ("create an embedding matrix for the words we have in the dataset")
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print ("A simple bidirectional LSTM with glove embeddings and two dense layers")

############ Model 2 ######################

input_cat = Input((len(cat_features_hash), ))
input_num = Input((len(num_features), ))
input_words = Input((maxlen, ))

x_num = Dense(100, activation="relu")(input_num)

x_cat = Embedding(max_size, 10)(input_cat)
x_cat = SpatialDropout1D(0.3)(x_cat)
x_cat = Flatten()(x_cat)
x_cat = Dense(100, activation="relu")(x_cat)

x_words = Embedding(max_features, 300, weights=[embedding_matrix], trainable=False)(input_words)
x_words = SpatialDropout1D(0.3)(x_words)
x_words = Bidirectional(GRU(50, return_sequences=True))(x_words)
x_words = Convolution1D(100, 3, activation="relu")(x_words)
x_words = GlobalMaxPool1D()(x_words)

x = concatenate([x_cat, x_num, x_words])
x = Dense(50, activation="relu")(x)
x = Dropout(0.25)(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])


############ Model 1 ####################33

# model1 = Sequential()
# model1.add(Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
# model1.add(SpatialDropout1D(0.3))
# model1.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))
# model1.add(Dense(1024, activation='relu'))
# model1.add(Dropout(0.5))
# model1.add(Dense(1024, activation='relu'))
# model1.add(Dropout(0.6))
# model1.add(Dense(1))
# model1.add(Activation('sigmoid'))
# model1.compile(loss='binary_crossentropy', optimizer='adam')


############# Combined Model ###############################

input_num = Input((24, ))
x_num = Dense(64, activation="relu")(input_num)
x_num = Dropout(0.2)(x_num)
x_num = Dense(32, activation="relu")(x_num)
x_num = Dropout(0.2)(x_num)

embed = Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=70, trainable=False)
sequence_input = Input(shape=(70,), dtype='int32')
embedded_sequences = embed(sequence_input)
lstm = LSTM(300, dropout=0.3, recurrent_dropout=0.3)(embedded_sequences)
lstmd = Dropout(0.4)(lstm)
dens1 = Dense(1024, activation='relu')(lstmd)
x_embd = Dropout(0.5)(dens1)

x = concatenate([x_num, x_embd])
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

print ("Fit the model with early stopping callback")
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
model = Model(inputs=[sequence_input, input_num], outputs=output)
model.compile(optimizer=optimizers.Adam(0.001, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
model.fit([xtrain_pad, train_feats], ytrain, validation_split=0.1, epochs=5, batch_size=128, callbacks=[earlystop])

################## Combined Model ###################################

# print ("Fit the model with early stopping callback")
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
# model1.fit(xtrain_pad, y=ytrain, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid), callbacks=[earlystop])

print ("predicting")
preds = model.predict([xtest_pad, testfeats])
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = preds
sample_submission.to_csv("sub/keras_baseline.csv", index=False)