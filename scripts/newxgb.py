from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sparse
import lightgbm as lgb
import pandas as pd 
import gc 

target_column = 'project_is_approved'
print "reading data"
id_column = 'id'
normdf = pd.read_csv("models/preprocessed/featured.csv")
Y = normdf[normdf['is_train'] == 1]['project_is_approved']
normdf.drop(['project_is_approved'], axis = 1)

test_id = normdf[normdf['is_test'] == 1]['id']

cat_feats =  ['project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'school_state', 'teacher_id', 'teacher_prefix']
for c in cat_feats:
	le = LabelEncoder()
	le.fit(normdf[c].astype(str))
	normdf[c] = le.transform(normdf[c].astype(str))
num_feats = ['teacher_number_of_previously_posted_projects', 'price', 'total_price', 'items', 'quantity', 'min_price',
'min_total_price', 'min_quantity', 'max_price', 'max_total_price',
'max_quantity', 'mean_price', 'mean_total_price', 'mean_quantity',
'avg_price', 'project_essay_1_pol', 'project_essay_1_sub',
'project_essay_1_word_count', 'project_essay_1_char_count',
'project_essay_1_density', 'project_essay_1_stopword',
'project_essay_1_punctuation_count',
'project_essay_1_upper_case_word_count',
'project_essay_1_title_word_count', 'project_essay_2_pol',
'project_essay_2_sub', 'project_essay_2_word_count',
'project_essay_2_char_count', 'project_essay_2_density',
'project_essay_2_stopword', 'project_essay_2_punctuation_count',
'project_essay_2_upper_case_word_count',
'project_essay_2_title_word_count', 'project_resource_summary_pol',
'project_resource_summary_sub', 'project_resource_summary_word_count',
'project_resource_summary_char_count',
'project_resource_summary_density',
'project_resource_summary_stopword',
'project_resource_summary_punctuation_count',
'project_resource_summary_upper_case_word_count',
'project_resource_summary_title_word_count',
'resource_description_pol', 'resource_description_sub',
'resource_description_word_count', 'resource_description_char_count',
'resource_description_density', 'resource_description_stopword',
'resource_description_punctuation_count',
'resource_description_upper_case_word_count',
'resource_description_title_word_count', 'project_title_pol',
'project_title_sub', 'project_title_word_count',
'project_title_char_count', 'project_title_density',
'project_title_stopword', 'project_title_punctuation_count',
'project_title_upper_case_word_count',
'project_title_title_word_count', 'year', 'dayofyear', 'weekday',
'day', 'month', 'hour', 'school_state_count', 'teacher_id_count',
'teacher_prefix_count', 'project_grade_category_count',
'project_subject_categories_count',
'project_subject_subcategories_count', 'year_count',
'dayofyear_count', 'weekday_count', 'day_count', 'month_count',
'hour_count']

relevant_columns = num_feats + cat_feats

normdf[relevant_columns] = normdf[relevant_columns].fillna(99)

std = RobustScaler()
normdf[relevant_columns] = pd.DataFrame(std.fit_transform(normdf[relevant_columns])).set_index(normdf.index)
train_features = normdf[normdf['is_train'] == 1]
test_features = normdf[normdf['is_test'] == 1]

print "reading vectors"
textColumns = ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'resource_description', 'project_title']

ts_vects = []
tr_vects = []
for col in textColumns:
	ts_vect = sparse.load_npz("models/vectors/new_"+col+".npz")
	# tr_vect = sparse.load_npz("models/vectors/new_tr_"+col+".npz")
	ts_vects.append(ts_vect)
	# ts_vects.append(ts_vect)

print "stacking vectors"
# tr_vects.append(csr_matrix(train_features[relevant_columns]))
# X_train_stack = hstack(tr_vects, 'csr')

ts_vects.append(csr_matrix(test_features[relevant_columns]))
X_test_stack = hstack(ts_vects, 'csr')
del ts_vects
gc.collect()

# params_lgb = {
#         'boosting_type': 'dart',
#         'objective': 'binary',
#         'metric': 'auc',
#         'max_depth': 10,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.25,
#         'bagging_fraction': 0.85,
#         'seed': 0,
#         'verbose': 0,
#     }

# print "Starting modelling"
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, Y, test_size=0.25, random_state=42)
# model = lgb.train(params_lgb, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
#         		  valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=100)
# model.save_model('models/newlgb1.txt')
# del model
# gc.collect()

model = lgb.Booster(model_file='models/newlgb1.txt')
test_preds = model.predict(X_test_stack, num_iteration=model.best_iteration)
sub = pd.DataFrame()
sub['id'] = test_id
sub[target_column] = test_preds
sub.to_csv("sub/newlgb.csv", index=False)