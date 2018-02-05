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

prepare_data = True 
feature_engg_flag = False
cleaning = True 
tf_idf_flag = True
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
	rc = pd.read_csv("data/resources.csv", index_col=id_column).fillna(missing_token)
	
	rc['total_price'] = rc.quantity * rc.price
	rc['price_sum'] = rc['price'].copy()
	rc['quantity_sum'] = rc['quantity'].copy()

	###################### Datasets Preparation #####################################
	print "dataset preparation"

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
	del test, train, rc, agg_rc

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

drop_columns = ['teacher_id', 'project_essay_1','project_essay_2','project_essay_3','project_essay_4' ,'project_subject_categories',"project_subject_subcategories", "project_resource_summary","project_title","description","project_submitted_datetime"]
if feature_engg_flag:
	print "feature engineering"
	df["Year"] = df["project_submitted_datetime"].dt.year
	df["Date of Year"] = df['project_submitted_datetime'].dt.dayofyear
	df["Weekday"] = df['project_submitted_datetime'].dt.weekday
	df["Day of Month"] = df['project_submitted_datetime'].dt.day
	df["Month"] = df['project_submitted_datetime'].dt.month
	df["Hour"] = df['project_submitted_datetime'].dt.hour

	df["essay1_len"] = df['project_essay_1'].apply(len)
	df["essay2_len"] = df['project_essay_2'].apply(len)
	df["essay3_len"] = df['project_essay_3'].apply(len)
	df["essay4_len"] = df['project_essay_4'].apply(len)
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

	df.drop(drop_columns, axis=1, inplace=True)
	df.to_csv("models/preprocessed/normdf.csv")

	del std

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
											str(row['project_essay_3']), str(row['project_essay_4'])]), axis=1)
	df['resource'] = df.apply(lambda row: ' '.join([str(row['description']), str(row['project_resource_summary'])]), axis=1)

	complete_text = df['text']
	title_text = df['project_title']
	summary = df['resource']
	# df.drop(drop_columns, axis=1, inplace=True)
	# df.to_csv('models/preprocessed/processed_input.csv')
	# exit(0)
	# del df



	# print "Fitting Word Vectors complete_text"
	# vect_word = TfidfVectorizer(max_features=8000, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
	# vect_word.fit(complete_text[traindex])
	
	# print "Transforming Word Vectors - Train"
	# tr_vect = vect_word.transform(complete_text[traindex]) 
	# sparse.save_npz('models/vectors/tr_vect_complete_text_cln.npz', tr_vect)

	# print "Transforming Word Vectors - Test"
	# ts_vect = vect_word.transform(complete_text[tesdex]) 
	# sparse.save_npz('models/vectors/ts_vect_complete_text_cln.npz', ts_vect)





	# print "Fitting Word Vectors title_text"
	# vect_word = TfidfVectorizer(max_features=3000, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
	# vect_word.fit(title_text[traindex])
	
	# print "Transforming Word Vectors - Train"
	# tr_vect = vect_word.transform(title_text[traindex]) 
	# sparse.save_npz('models/vectors/tr_vect_title_text_cln.npz', tr_vect)

	# print "Transforming Word Vectors - Test"
	# ts_vect = vect_word.transform(title_text[tesdex]) 
	# sparse.save_npz('models/vectors/ts_vect_title_text_cln.npz', ts_vect)






	print "Fitting Word Vectors summary + desc "
	vect_word = TfidfVectorizer(max_features=6000, analyzer='word', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
	vect_word.fit(summary[traindex])
	
	print "Transforming Word Vectors - Train"
	tr_vect = vect_word.transform(summary[traindex]) 
	sparse.save_npz('models/vectors/tr_vect_summary_desc_cln.npz', tr_vect)

	print "Transforming Word Vectors - Test"
	ts_vect = vect_word.transform(summary[tesdex]) 
	sparse.save_npz('models/vectors/ts_vect_summary_desc_cln.npz', ts_vect)




	# print "Fitting Character Vectors"
	# vect_char = TfidfVectorizer(max_features=2500, analyzer='char', stop_words='english', ngram_range=(1,3), dtype=np.float32) 
	# vect_char.fit(complete_text[traindex] + summary[traindex])

	# print "Transforming Character Vectors - Train"
	# tr_vect_char = vect_char.transform(complete_text[traindex]) 
	# sparse.save_npz('models/vectors/tr_vect_char_sm.npz', tr_vect_char)

	# print "Transforming Character Vectors - Test"
	# ts_vect_char = vect_char.transform(complete_text[tesdex]) 
	# sparse.save_npz('models/vectors/ts_vect_char_sm.npz', ts_vect_char)