from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sparse
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
import numpy as np 
import pandas as pd 
import warnings
import time
import gc
import os

warnings.filterwarnings('ignore')
start = time.time()

feature_engg_flag = False
tf_idf_flag = False
model_flag = True
read_data = False
seed = 23

def create_feature_map(features):
    outfile = open('models/xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

####################### Read Data ###############################################
print "reading data"

id_column = "id"
missing_token = " UNK "
target_column = "project_is_approved"

if read_data:
    train = pd.read_csv("data/train.csv", index_col=id_column, parse_dates=["project_submitted_datetime"])
    test = pd.read_csv("data/test.csv", index_col=id_column, parse_dates=["project_submitted_datetime"])
    sub = pd.read_csv("data/resources.csv")

    print "dataset preparation"
    Y = train[target_column].copy()

    traindex = train.index
    tesdex = test.index
    del test, train


def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        fscore = clf.booster().get_fscore()
    else:
        fscore = clf.get_fscore()

    feat_importances = []
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)

    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    print feat_importances
    return feat_importances

if model_flag:
    # normdf = pd.read_csv("models/preprocessed/normdf.csv", index_col=id_column)
    # sentidf = pd.read_csv("models/preprocessed/sentiment.csv", index_col=id_column)
    # nlpdf = pd.read_csv("models/preprocessed/nlpfeats.csv", index_col=id_column)
    # normdf = pd.merge(normdf, sentidf, left_index=True, right_index=True, how="inner")
    # normdf = pd.merge(normdf, nlpdf, left_index=True, right_index=True, how="inner")
    
    # drop_cols = ['Hour','project_grade_category', 'Count_project_grade_category','teacher_prefix', 'Count_teacher_prefix', 'Year', 'Count_teacher_id', 'Weekday', 'school_state']
    # normdf = normdf.drop(drop_cols, axis=1)

    # new features 

    print "reading data"
    id_column = 'id'
    normdf = pd.read_csv("models/preprocessed/featured.csv")
    Y = normdf[normdf['is_train'] == 1]['project_is_approved']
    normdf.drop(['project_is_approved'], axis = 1)

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

    # PolynomialFeatures(normdf[num_feats])

    relevant_columns = num_feats + cat_feats
    normdf[relevant_columns] = normdf[relevant_columns].fillna(99)

    std = RobustScaler()
    normdf[relevant_columns] = pd.DataFrame(std.fit_transform(normdf[relevant_columns])).set_index(normdf.index)
    traindf = normdf[normdf['is_train'] == 1]
    X_train_stack = traindf[relevant_columns]

    print "Loading Vectors"
    # tr_vect1 = sparse.load_npz("models/vectors/tr_vect_complete_text.npz")
    # tr_vect2 = sparse.load_npz("models/vectors/tr_vect_title_text.npz")
    # tr_vect3 = sparse.load_npz("models/vectors/tr_vect_summary.npz")

    # tr_vect_char = sparse.load_npz("models/vectors/tr_vect_char.npz")

    # ts_vect1 = sparse.load_npz("models/vectors/ts_vect_complete_text.npz")
    # ts_vect2 = sparse.load_npz("models/vectors/ts_vect_title_text.npz")
    # ts_vect3 = sparse.load_npz("models/vectors/ts_vect_summary.npz")
    # ts_vect_char = sparse.load_npz("models/vectors/ts_vect_char.npz")
    # Y = Y[traindex]

    # X_train_stack = normdf.loc[traindex]
    # X_train_stack = hstack([csr_matrix(normdf.loc[traindex,])], 'csr')
    # X_train_stack = normdf.loc[traindex,]
    # del tr_vect1,tr_vect2,tr_vect3
    # gc.collect()

    # X_test_stack = normdf.loc[tesdex]
    # X_test_stack = hstack([csr_matrix(normdf.loc[tesdex,])], 'csr')
    # X_test_stack = normdf.loc[tesdex,]
    # del ts_vect1, ts_vect2, ts_vect3
    # gc.collect()

    # del normdf
    # gc.collect()

    ################## Modelling ##########################
    print "Modelling"

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, Y, test_size=0.25, random_state=seed)
    
    xgb_params = {'eta': 0.1, 
                  'n_estimators' : 1000,
                  'max_depth': 12,
                  'min_child_weight' : 1,
                  'gaama' : 0, 
                  'subsample': 0.8, 
                  'colsample_bytree': 0.8,
                  'scale_pos_weight' : 1,

                  'nthread' : 4, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': seed
                 }

    xgb_params = {
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.25,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 0,
        'silent': 1,
    }

    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    # d_test = xgb.DMatrix(X_test_stack)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 250, watchlist, verbose_eval=10, early_stopping_rounds=20)    

    # param_test1 = {'learning_rate':[0.01, 0.1, 0.3]}
    # gsearch = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate=0.1, 
    #     n_estimators=150, max_depth=10, min_child_weight=1, gamma=0, subsample=0.8, 
    #     colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
    #     param_grid = param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # gsearch.fit(X_train_stack, Y)
    # print gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    # model = gsearch

    get_xgb_feat_importances(model)
    del X_train, X_valid, y_train, y_valid; gc.collect()

    ################## Prediction #########################
    # print "Predicting"

    # xgb_pred = model.predict(d_test)
    # sub = pd.DataFrame(xgb_pred, columns=[target_column], index=tesdex)
    # sub.to_csv("sub/xgb_feats_baseline.csv",index=True)
    
    ################## Complete ###########################