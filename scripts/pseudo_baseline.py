from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.preprocessing import PolynomialFeatures
import scipy.sparse as sparse
import lightgbm as lgb
import pandas as pd 
import gc 
import xgboost as xgb

target_column = "project_is_approved"
id_column = 'id'

print "dataset preparation"
normdf = pd.read_csv("models/preprocessed/featured_2.csv")
submitted = pd.read_csv('sub/ens_submission3.csv')
submitted[target_column] = submitted[target_column].apply(lambda x : 1 if x >= 0.5 else 0)

testid = normdf[normdf['is_test'] == 1][id_column]
Y = normdf[normdf['is_train'] == 1][target_column]
Y_test = normdf[normdf['is_test']==1][target_column]
Y_test = submitted[target_column]
Y_combined = list(Y) + list(Y_test)
normdf[target_column] = Y_combined
Y = normdf[target_column]

drop_cols = [target_column, id_column]
normdf = normdf.drop(drop_cols, axis = 1)

print "Label Encoding"
cat_feats =  ['project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'school_state', 'teacher_id', 'teacher_prefix']
for c in cat_feats:
    le = LabelEncoder()
    le.fit(normdf[c].astype(str))
    normdf[c] = le.transform(normdf[c].astype(str))

    ### pseudo adding
    # if c in pseudo.columns:
	   #  pseudo[c] = le.transform(pseudo[c].astype(str))


relevant_cols = """teacher_id
project_essay_2_density
project_resource_summary_density
resource_description_density
project_essay_1_sub
project_essay_2_sub
project_essay_2_pol
project_essay_1_density
dayofyear_count
dayofyear
project_essay_2_char_count
project_essay_1_pol
total_price
price
project_essay_1_char_count
project_title_density
project_essay_2_stopword
project_essay_2_punctuation_count
min_price
min_total_price
project_subject_subcategories_count
resource_description_sub
project_resource_summary_char_count
resource_description_pol
project_essay_2_word_count
max_price"""
relevant_cols = [x for x in relevant_cols.split("\n")]

print "Robust Scaling"
normdf = normdf.fillna(99)
std = RobustScaler()
normdf[relevant_cols] = pd.DataFrame(std.fit_transform(normdf[relevant_cols])).set_index(normdf.index)
# pseudo[relevant_cols] = pd.DataFrame(std.fit_transform(pseudo[relevant_cols])).set_index(pseudo.index)

predict = True

print "Loading Vectors"
textColumns = ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'resource_description', 'project_title']
tr_vects = []
ts_vects = []
ps_vects = []
for i, col in enumerate(textColumns):
    if predict == False:
        tr_vect = sparse.load_npz("models/vectors/new_tr_"+col+".npz")
        # print tr_vect.shape
        tr_vects.append(tr_vect)

        ############### pseudo data ###################
        ts_vect = sparse.load_npz("models/vectors/new_"+col+".npz")        
        # print ts_vect.shape
        ts_vects.append(ts_vect)


        # ps_vect = sparse.load_npz("models/vectors/new_ps_"+col+".npz")
        # ps_vects.append(ps_vect)
        ############### pseudo data ###################

        # tr_vect = sparse.load_npz("models/vectors/new_char_tr_"+col+".npz")
        # tr_vects.append(tr_vect)

    else:
        ts_vect = sparse.load_npz("models/vectors/new_"+col+".npz")        
        ts_vects.append(ts_vect)

print "Stacking Vectors"
if predict == False:
    traindf = normdf[normdf['is_train'] == 1][relevant_cols]
    tr_vects.append(csr_matrix(traindf))
    X_train_stack = hstack(tr_vects, 'csr')

    print "stacking pseudo"
    ################ pseudo data adding ####################
    
    # pseudoY = pseudo[target_column]
    # Y = Y.append(pseudoY)
    # pseudodf = pseudo[relevant_cols]

    pseudodf = normdf[normdf['is_test'] == 1][relevant_cols]
    ts_vects.append(csr_matrix(pseudodf))
    X_pseudo_stack = hstack(ts_vects, 'csr')

    X_train_stack = vstack([X_train_stack, X_pseudo_stack])
    ############### Added Pseudo Data #######################

    # X_train_stack = new_traindf
    del tr_vects, normdf, traindf

else:
    testdf = normdf[normdf['is_test'] == 1][relevant_cols]

    ts_vects.append(csr_matrix(testdf))
    X_test_stack = hstack(ts_vects, 'csr')

    del ts_vects, normdf, testdf
gc.collect()


if predict:
    ############## prediction
    
    model = lgb.Booster(model_file='models/uplgb_pseudo_comp.txt')
    test_preds = model.predict(X_test_stack, num_iteration=model.best_iteration)
    sub = pd.DataFrame()
    sub['id'] = testid
    sub[target_column] = test_preds
    sub.to_csv("sub/uplgb_pseudo_comp.csv",index=False)
    exit(0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, Y, test_size=0.25, random_state=42)

################### feature importance #####################
feat_imp = False
if feat_imp:
    def get_xgb_feat_importances(clf):
        if isinstance(clf, xgb.XGBModel):
            fscore = clf.booster().get_fscore()
        else:
            fscore = clf.get_fscore()

        feat_importances = []
        for ft, score in fscore.iteritems():
            feat_importances.append({'Feature': ft, 'Importance': score})
        feat_importances = pd.DataFrame(feat_importances)
        feat_importances = feat_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        feat_importances['Importance'] /= feat_importances['Importance'].sum()
        feat_importances.to_csv('models/feats.csv')
        return feat_importances

    xgb_params = {'eta': 0.1, 
                  'n_estimators' : 400,
                  'max_depth': 8,
                  'min_child_weight' : 1,
                  'gaama' : 0, 
                  'subsample': 0.8, 
                  'colsample_bytree': 0.8,
                  'scale_pos_weight' : 1,
                  'nthread' : 4, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': 12 }

    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 250, watchlist, verbose_eval=10, early_stopping_rounds=20)    
    get_xgb_feat_importances(model)
    print "completed"
    del X_train, X_valid, y_train, y_valid; gc.collect()

    exit(0)
#########################################################################################################

print "modelling"

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

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=1000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=120, verbose_eval=10)
model1.save_model('models/uplgb_pseudo_comp.txt')
del model1
gc.collect()
exit(0)

"""
params1 = {
        'objective': 'binary',
        'metric': 'auc',

        'boosting_type': 'dart',
        'max_depth': 10, 
        # 'num_leaves': 31, 
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

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb2.txt')
del model1
gc.collect()

params1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16, 
        'num_leaves': 31, 
        'learning_rate': 0.005, 
        'feature_fraction': 0.75, 
        'bagging_fraction': 0.75, 
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':42
}  

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb3.txt')
del model1
gc.collect()

params1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 10, 
        'num_leaves': 32, 
        'learning_rate': 0.005, 
        'feature_fraction': 0.75, 
        'bagging_fraction': 0.25, 
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':42
}  

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb4.txt')
del model1
gc.collect()

params1 = {
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 18, 
        'num_leaves': 36, 
        'learning_rate': 0.05, 
        'feature_fraction': 0.85, 
        'bagging_fraction': 0.85, 
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':42
}  

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb5.txt')
del model1
gc.collect()

params1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 12, 
        'num_leaves': 56, 
        'learning_rate': 0.01, 
        'feature_fraction': 0.90, 
        'bagging_fraction': 0.35, 
        'bagging_freq': 3,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':42
}  

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb6.txt')
del model1
gc.collect()

params1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16, 
        'num_leaves': 31, 
        'learning_rate': 0.025, 
        'feature_fraction': 0.99, 
        'bagging_fraction': 0.99, 
        'bagging_freq': 8,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':427
}  

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb7.txt')
del model1
gc.collect()

params1 = {
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 14, 
        'num_leaves': 31, 
        'learning_rate': 0.001, 
        'feature_fraction': 0.85, 
        'bagging_fraction': 0.85, 
        'bagging_freq': 7,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':142
}  

model1 = lgb.train(params1, lgb.Dataset(X_train, y_train), num_boost_round=10000, 
        valid_sets=[lgb.Dataset(X_valid, y_valid)], early_stopping_rounds=200, verbose_eval=10)
model1.save_model('models/uplgb8.txt')
del model1
gc.collect()
"""




# today 
# start using character n-grams in your best model # not working very well 
# polynomial - not working very well 

# current model 
# stacking 8 lgb versions of baseline -> improved a little bit 
# ensembling with neural network 
# idea 3 : pseudo labelling -> very less improvement

# idea 1 : improve deep learning - 50%,50% splitting, generating new features 
# idea 2 : improve neural network - increase it to 79+

# idea 4 : generate more features - readability, anything else 
# idea 5 : use external data 