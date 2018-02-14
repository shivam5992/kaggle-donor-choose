from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from textstat.textstat import textstat
from scipy.stats import pearsonr
from scipy.sparse import hstack
from textblob import TextBlob	
import scipy.sparse as sparse
import string, time
import pandas as pd 
import numpy as np 
import re, gc, os
import warnings
import pickle

punctuation = string.punctuation
warnings.filterwarnings('ignore')
start = time.time()

stop_words = open('data/stopwords.txt').read().strip().split("\n")
stop_words = [x.replace("\r","") for x in stop_words]

id_column = "id"
missing_token = " UNK "
target_column = "project_is_approved"

print "reading data"
train = pd.read_csv("data/train.csv", parse_dates=["project_submitted_datetime"])
test = pd.read_csv("data/test.csv", parse_dates=["project_submitted_datetime"])
rc = pd.read_csv("data/resources.csv").fillna(missing_token)

train['is_train'] = 1
train['is_test'] = 0
test['is_train'] = 0
test['is_test'] = 1
test[target_column] = 99

print "combine train and test"
df = pd.concat((train, test))
del train, test
gc.collect()

print "adjust project_essay 2 and 4 according to the requirement"
df.loc[df.project_essay_4.isnull(), ['project_essay_4','project_essay_2']] = df.loc[df.project_essay_4.isnull(), ['project_essay_2','project_essay_4']].values
df[['project_essay_2','project_essay_3']] = df[['project_essay_2','project_essay_3']].fillna('')
df['project_essay_1'] = df.apply(lambda row: ' '.join([str(row['project_essay_1']), str(row['project_essay_2'])]), axis=1)
df['project_essay_2'] = df.apply(lambda row: ' '.join([str(row['project_essay_3']), str(row['project_essay_4'])]), axis=1)
df = df.drop(['project_essay_3', 'project_essay_4'], axis=1)
rc['total_price'] = rc['quantity']*rc['price']

print "generating new features - count, sum"
agg_rc = rc.groupby('id').agg({'description':'count', 'quantity':'sum', 'price':'sum', 'total_price':'sum'}).rename(columns={'description':'items'})

print "generating new features - min, max, mean"
for func in ['min', 'max', 'mean']:
	agg_rc_temp = rc.groupby('id').agg({'quantity':func, 'price':func, 'total_price':func}).rename(columns={'quantity':func+'_quantity', 'price':func+'_price', 'total_price':func+'_total_price'}).fillna(0)
	agg_rc = agg_rc.join(agg_rc_temp)
	del agg_rc_temp
	gc.collect()

print "combine descriptions"
agg_rc = agg_rc.join(rc.groupby('id').agg({'description':lambda x:' '.join(x.values.astype(str))}).rename(columns={'description':'resource_description'}))
agg_rc['avg_price'] = agg_rc.total_price / agg_rc.quantity

df = df.join(agg_rc, on='id')

### fix this 
pseudo = pd.read_csv('models/preprocessed/pseudo.csv')
df['is_pseudo'] = df['id'].apply(lambda x: 1 if x in list(pseudo['id']) else 0)

del agg_rc
gc.collect();

def get_sentiment(s):
	s = str(s).decode('ascii', 'ignore')
	sent = TextBlob(s).sentiment
	return str(sent.polarity) +"|"+ str(sent.subjectivity)

print "generating text features"
textColumns = ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'resource_description', 'project_title']
for col in textColumns:
	df[col] = df[col].fillna(missing_token)
	df[col+'_sentiment'] = df[col].apply(get_sentiment)
	df[col+'_pol'] = df[col+'_sentiment'].apply(lambda x : x.split("|")[0])
	df[col+'_sub'] = df[col+'_sentiment'].apply(lambda x : x.split("|")[1])
	df = df.drop([col+'_sentiment'], axis=1)

	df[col+"_word_count"] = df[col].apply(lambda x: len(x.split()))
	df[col+"_char_count"] = df[col].apply(len)
	df[col+'_density'] = df[col+'_char_count'] / (df[col+'_word_count']+1)

	df[col+'_stopword'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))
	df[col+'_punctuation_count'] = df[col].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
	df[col+'_upper_case_word_count'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
	df[col+'_title_word_count'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

print "generating date features"
df["year"] = df["project_submitted_datetime"].dt.year
df["dayofyear"] = df['project_submitted_datetime'].dt.dayofyear
df["weekday"] = df['project_submitted_datetime'].dt.weekday
df["day"] = df['project_submitted_datetime'].dt.day
df["month"] = df['project_submitted_datetime'].dt.month
df["hour"] = df['project_submitted_datetime'].dt.hour

print "creating count features"
features_for_count = ['school_state', 'teacher_id', 'teacher_prefix', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']
features_for_count += ['year', 'dayofyear', 'weekday', 'day', 'month', 'hour']
for col in features_for_count:
	aggDF = df.groupby(col).agg('count').rename(columns={'quantity':func+'_quantity'})
	aggDF[col] = aggDF.index
	tempDF = pd.DataFrame(aggDF[['project_submitted_datetime', col]], columns = ['project_submitted_datetime', col])
	tempDF = tempDF.rename(columns={'project_submitted_datetime': col+"_count"})
	df = df.merge(tempDF, on=col, how='left')

print "drop columns and write file"
drop_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary','resource_description', 'project_submitted_datetime']
df.drop(drop_columns, axis=1, inplace=True)
df.to_csv("models/preprocessed/featured.csv", index=False)

def cleanup_text(x):
	x = x.replace("\\r", " ").replace("\\t", " ").replace("\\n", " ")
	x = "".join(_ for _ in x if _ not in punctuation)
	return x.lower() 

def getTextFeatures(df, col, max_features, save = True):
	df[col] = df[col].fillna(missing_token)
	df[col] = df[col].apply(lambda x : cleanup_text(x))

	################### save vectors #######################
	if save:
		print "writing vectors"
		vect_word = TfidfVectorizer(max_features = max_features, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
		vect_word.fit(df[col])
		with open('models/vect_word_'+col+'.pk', 'wb') as fin:
			print "saving", col
			pickle.dump(vect_word, fin)
		################### loading pre-trained-vectors #####################
	else:
		vect_word = pickle.load(open("models/vect_word_"+col+".pk", "rb"))

	print col, "train"
	tr_vect = vect_word.transform(df[df['is_train']==1][col]) 
	sparse.save_npz('models/vectors/new_char_tr_'+col+'.npz', tr_vect)

	print col, "test"
	ts_vect = vect_word.transform(df[df['is_test' ]==1][col]) 
	sparse.save_npz('models/vectors/new_char_'+col+'.npz', ts_vect)

	print col, "pseudo"
	ps_vect = vect_word.transform(df[df['is_pseudo' ]==1][col]) 
	sparse.save_npz('models/vectors/new_ps_'+col+'.npz', ps_vect)



textColumns = ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'resource_description', 'project_title']
# max_feats = [2000, 2000, 500, 600, 200] # chars
max_feats = [5000, 6000, 1000, 1000, 500] # words 
for i, col in enumerate(textColumns):
	getTextFeatures(df, col, max_features=max_feats[i], save=False)


# https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data
# https://research.donorschoose.org/t/download-opendata/33/8