from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import scipy.sparse as sparse
import lightgbm as lgb
import pandas as pd 
import gc 

id_column = 'id'
target_column = "project_is_approved"
seed = 123 

print "dataset preparation"
train = pd.read_csv("data/train.csv", index_col=id_column)
test = pd.read_csv("data/test.csv", index_col=id_column)

normdf = pd.read_csv("models/preprocessed/normdf.csv", index_col=id_column)
sentidf = pd.read_csv("models/preprocessed/sentiment.csv", index_col=id_column)
nlpdf = pd.read_csv("models/preprocessed/nlpfeats.csv", index_col=id_column)

normdf = pd.merge(normdf, sentidf, left_index=True, right_index=True, how="inner")
normdf = pd.merge(normdf, nlpdf, left_index=True, right_index=True, how="inner")

drop_cols = ['Hour','project_grade_category', 'Count_project_grade_category','teacher_prefix', 'Count_teacher_prefix', 'Year', 'Count_teacher_id', 'Weekday', 'school_state']
normdf = normdf.drop(drop_cols, axis=1)

traindex = train.index
tesdex = test.index
Y = train[target_column].copy()
Y = Y[traindex]
del train, test 
gc.collect()


print "Loading Vectors"
tr_vect1 = sparse.load_npz("models/vectors/tr_vect_complete_text_cln.npz")
tr_vect2 = sparse.load_npz("models/vectors/tr_vect_title_text_cln.npz")
tr_vect3 = sparse.load_npz("models/vectors/tr_vect_summary_desc_cln.npz")
X_train_stack = hstack([tr_vect1, tr_vect2, tr_vect3, csr_matrix(normdf.loc[traindex,])], 'csr')
del tr_vect1,tr_vect2,tr_vect3
gc.collect()

ts_vect1 = sparse.load_npz("models/vectors/ts_vect_complete_text_cln.npz")
ts_vect2 = sparse.load_npz("models/vectors/ts_vect_title_text_cln.npz")
ts_vect3 = sparse.load_npz("models/vectors/ts_vect_summary_desc_cln.npz")
X_test_stack = hstack([ts_vect1, ts_vect2, ts_vect3, csr_matrix(normdf.loc[tesdex,])], 'csr')
del ts_vect1, ts_vect2, ts_vect3
gc.collect()

del normdf
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, Y, test_size=0.25, random_state=42)


params1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16, 
        'num_leaves': 31, 
        'learning_rate': 0.025, 
        'feature_fraction': 0.75, 
        'bagging_fraction': 0.75, 
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':42
}  
print "model1"
model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/lgb1.txt')
del model1
gc.collect()

# model = lgb.Booster(model_file='models/lgb1.txt')
# test_preds = model.predict(X_test_stack, num_iteration=model.best_iteration)
# sub = pd.DataFrame(test_preds, columns=[target_column], index=tesdex)
# sub.to_csv("sub/lgb1_test.csv",index=True)


# today 
# idea 1 : start using deep learning - 50%,50% splitting, generating new features 
# idea 2 : start using character n-grams in your best model 

# later 
# idea 3 : generate more features - readability, polynomial
# idea 4 : pseudo labelling 