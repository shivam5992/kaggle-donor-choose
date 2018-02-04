from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sparse
from collections import Counter 
from sklearn.grid_search import GridSearchCV
from textblob import TextBlob
	

import numpy as np 
import pandas as pd 
import warnings
import time
import gc
import os
import string
import cPickle


punctuation = string.punctuation
warnings.filterwarnings('ignore')
start = time.time()

stop_words = open('data/stopwords.txt').read().strip().split("\n")
stop_words = [x.replace("\r","") for x in stop_words]
# stop_words = set(stopwords.words('english'))

prepare_data = True 
feature_engg_flag = True
cleaning = False 
tf_idf_flag = False
model_flag = False 
predicting = False 

id_column = "id"
missing_token = " UNK "
target_column = "project_is_approved"

####################### Read Data ###############################################
if prepare_data:
	print "reading data"

	train = pd.read_csv("data/train.csv", index_col=id_column, parse_dates=["project_submitted_datetime"])
	test = pd.read_csv("data/test.csv", index_col=id_column, parse_dates=["project_submitted_datetime"])
	
	# rc = pd.read_csv("data/resources.csv", index_col=id_column).fillna(missing_token)
	# rc['total_price'] = rc.quantity * rc.price
	# rc['price_sum'] = rc['price'].copy()
	# rc['quantity_sum'] = rc['quantity'].copy()
	# rc['quantity_count'] = rc['quantity'].copy()

	###################### Datasets Preparation #####################################
	print "dataset preparation"

	Y = train[target_column].copy()

	# train = train.drop(target_column, axis=1) # axis:1 drop in all rows (vertical drop)
	df = pd.concat([train, test], axis=0) # axis:0 concatenate all columns together


	### add count as well 
	# aggregate resources data frame 
	# agg_rc = rc.reset_index().groupby(id_column).agg(dict(quantity_count='count', price_sum='sum', quantity_sum='sum', total_price='mean', quantity='mean', price='mean', description=lambda x: missing_token.join(x)))
	# merge resources data frame with input dataframe
	# df = pd.merge(df, agg_rc, left_index=True, right_index=True, how= "inner")

	traindex = train.index
	tesdex = test.index
	# alldex = df.index
	del test, train#, rc, agg_rc

#################### Feature Engineering ########################################

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

def get_polarity(text):
	try:
		textblob = TextBlob(text)
		pol = textblob.sentiment.polarity
	except Exception as E:
		pol = 0.0
	return pol

def get_subjectivity(text):
	try:
		textblob = TextBlob(text)
		subj = textblob.sentiment.subjectivity
	except:
		subj = 0.0
	return subj

pos_dic = {
	'noun' : ['NN','NNS','NNP','NNPS'],
	'pron' : ['PRP','PRP$','WP','WP$'],
	'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
	'adj' :  ['JJ','JJR','JJS'],
	'adv' : ['RB','RBR','RBS','WRB']
}

def pos_check(x, flag):
	cnt = 0
	try:
		wiki = TextBlob(x)
		for tupe in wiki.tags:
			ppo = list(tupe)[1]
			if ppo in pos_dic[flag]:
				cnt += 1
	except Exception as E:
		pass
	return cnt 



drop_columns = ['teacher_id', 'project_essay_1','project_essay_2','project_essay_3','project_essay_4' , "project_resource_summary","project_title","description","project_submitted_datetime"]
if feature_engg_flag:
	print "feature engineering"
	# df["Year"] = df["project_submitted_datetime"].dt.year
	# df["Date of Year"] = df['project_submitted_datetime'].dt.dayofyear
	# df["Weekday"] = df['project_submitted_datetime'].dt.weekday
	# df["Day of Month"] = df['project_submitted_datetime'].dt.day
	# df["Month"] = df['project_submitted_datetime'].dt.month
	# df["Hour"] = df['project_submitted_datetime'].dt.hour

	# df["essay1_len"] = df['project_essay_1'].apply(len)
	# df["essay2_len"] = df['project_essay_2'].apply(len)
	
	# df['project_essay_3'] = df['project_essay_3'].fillna(' ')
	# df['project_essay_4'] = df['project_essay_4'].fillna(' ')

	# df["essay3_len"] = df['project_essay_3'].apply(len)
	# df["essay4_len"] = df['project_essay_4'].apply(len)
	# df["title_len"] = df['project_title'].apply(len)

	df['temp_text'] = df.apply(lambda row: ' '.join([str(row['project_essay_1']), str(row['project_essay_2']), 
										str(row['project_essay_3']), str(row['project_essay_4'])]), axis=1)
	#### some more features ## 
	df['word_count'] = df['temp_text'].apply(lambda x: len(x.split()))
	df['char_count'] = df['temp_text'].apply(len)
	df['word_density'] = df['char_count'] / (df['word_count']+1)
	df['stopword_count'] = df['temp_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))
	df['punctuation_count'] = df['temp_text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
	df['upper_case_word_count'] = df['temp_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
	df['title_word_count'] = df['temp_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
	
	# df['noun_count'] = df['temp_text'].apply(lambda x: pos_check(x, 'noun'))
	# print "9"
	# df['verb_count'] = df['temp_text'].apply(lambda x: pos_check(x, 'verb'))
	# print "10"
	# df['adj_count'] = df['temp_text'].apply(lambda x: pos_check(x, 'adj'))
	# print "11"
	# df['adv_count'] = df['temp_text'].apply(lambda x: pos_check(x, 'adv'))
	# print "12"
	# df['pron_count'] = df['temp_text'].apply(lambda x: pos_check(x, 'pron'))
	
	nlp_feats = ['word_count', 'char_count', 'word_density', 'stopword_count', 'punctuation_count', 'upper_case_word_count', 'title_word_count']
	df[nlp_feats].to_csv("models/preprocessed/nlpfeats.csv")
	exit(0)

	df['sent_polarity'] = df['temp_text'].apply(lambda x: get_polarity(x))
	df['sent_subjectivity'] = df['temp_text'].apply(lambda x: get_subjectivity(x))
	df[['sent_polarity', 'sent_subjectivity']].to_csv("models/preprocessed/sentiment.csv")

	# df["sentiment"] = df['temp_text'].apply(lambda x: get_polarity(x.encode("utf8").decode("ascii","ignore")))
	# df["polarity"] = df['temp_text'].apply(lambda x: get_subjectivity(x.encode("utf8").decode("ascii","ignore")))

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

	std = StandardScaler()
	keep_columns = list(set(list(df.columns)) - set(drop_columns))
	df[keep_columns] = pd.DataFrame(std.fit_transform(df[keep_columns])).set_index(alldex)
	drop_columns.append('temp_text')

	df.drop(drop_columns, axis=1, inplace=True)
	df.to_csv("models/preprocessed/normdf1.csv")

	del std
exit(0)

if tf_idf_flag:
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
	complete_text = df['text']
	df.drop(drop_columns, axis=1, inplace=True)
	# df.to_csv('models/preprocessed/processed_input.csv')
	# exit(0)
	del df

	print "Fitting Word Vectors"
	vect_word = TfidfVectorizer(max_features=15000, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
	vect_word.fit(complete_text[traindex])
	
	print "Transforming Word Vectors - Train"
	tr_vect = vect_word.transform(complete_text[traindex]) 
	sparse.save_npz('models/vectors/tr_vect_cln.npz', tr_vect)

	print "Transforming Word Vectors - Test"
	ts_vect = vect_word.transform(complete_text[tesdex]) 
	sparse.save_npz('models/vectors/ts_vect_cln.npz', ts_vect)

	print "Fitting Character Vectors"
	vect_char = TfidfVectorizer(max_features=6000, analyzer='char', stop_words='english', ngram_range=(2,4), dtype=np.float32) 
	vect_char.fit(complete_text[traindex])

	print "Transforming Character Vectors - Train"
	tr_vect_char = vect_char.transform(complete_text[traindex]) 
	sparse.save_npz('models/vectors/tr_vect_char_cln.npz', tr_vect_char)

	print "Transforming Character Vectors - Test"
	ts_vect_char = vect_char.transform(complete_text[tesdex]) 
	sparse.save_npz('models/vectors/ts_vect_char_cln.npz', ts_vect_char)

	del vect_word, vect_char, complete_text
	gc.collect()

if model_flag:
	train = pd.read_csv("data/train.csv", index_col=id_column)
	Y = train[target_column].copy()

	traindex = train.index 
	del train
	gc.collect()

	normdf = pd.read_csv("models/preprocessed/normdf.csv", index_col=id_column)

	print "Loading Vectors"
	tr_vect = sparse.load_npz("models/vectors/tr_vect_cln.npz")
	tr_vect_char = sparse.load_npz("models/vectors/tr_vect_char_cln.npz")

	train_features = hstack([tr_vect, tr_vect_char, csr_matrix(normdf.loc[traindex,])], 'csr')
	del tr_vect, tr_vect_char
	gc.collect()

	################## Modelling ##########################
	print "Modelling"
	
	lr = LogisticRegression(solver="sag")

	# param_grid = {'C' : [2, 3, 4]}
	# gsearch = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='log_loss', n_jobs=16, cv=5)
	# gsearch.fit(train_features, Y)
	# print gsearch.best_params_
	# print gsearch.best_score_

	lr.fit(train_features, Y)
	
	with open('models/logit.pkl', 'wb') as fid:
		cPickle.dump(lr, fid)    
	print("Auc Score: ", np.mean(cross_val_score(lr, train_features, Y, cv=3, scoring='roc_auc')))

################## Prediction #########################
if predicting:
	print "Predicting"
	normdf = pd.read_csv("models/preprocessed/normdf.csv", index_col=id_column)

	test = pd.read_csv("data/test.csv", index_col=id_column)
	tesdex = test.index

	with open('models/logit.pkl', 'rb') as fid:
		lr = cPickle.load(fid)

		ts_vect = sparse.load_npz("models/vectors/ts_vect_cln.npz")
		ts_vect_char = sparse.load_npz("models/vectors/ts_vect_char_cln.npz")
		test_features = hstack([ts_vect, ts_vect_char, csr_matrix(normdf.loc[tesdex,])], 'csr')
		del ts_vect, ts_vect_char, normdf
		gc.collect()

		sub = pd.DataFrame(lr.predict_proba(test_features)[:, 1], columns=[target_column],index=tesdex)
		sub.to_csv("sub/logistic_sub1.csv",index=True)

	################## Complete ###########################