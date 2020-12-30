import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor

# create the Labelencoder object
le = preprocessing.LabelEncoder()
df = pd.read_csv('data/patient_regression_analysis_v3.csv',sep=",")
df["primaryid"] = df["primaryid"].astype(str)
#convert the categorical columns into numeric
df['primaryid'] = le.fit_transform(df['primaryid'])
df['i_f_code'] = le.fit_transform(df['i_f_code'])
df['role_cod'] = le.fit_transform(df['role_cod'])
df['rept_cod'] = le.fit_transform(df['rept_cod'])
df['drug_name'] = le.fit_transform(df['drug_name'])
df['pt'] = le.fit_transform(df['pt'])
df['indi_pt'] = le.fit_transform(df['indi_pt'])
df['sex'] = le.fit_transform(df['sex'])
df['outc_cod'] = le.fit_transform(df['outc_cod'])

X = df.drop('Days', axis=1).values 
y = df['Days'].values




def baseline_regression(train_dat,train_response,test_dat,test_response):
    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr.fit(train_dat,train_response)
    pred_result = dummy_regr.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse

def linear_regression(train_dat,train_response,test_dat,test_response):
    LR = LinearRegression()
    LR.fit(train_dat,train_response)
    pred_result = LR.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse

def ridge_regression(train_dat,train_response,test_dat,test_response):
    ridge_reg = Ridge()
    param = {'alpha':[1e-5,1e-2,1e-1,1,5]}
    ridge_regress = GridSearchCV(ridge_reg,param,scoring='neg_mean_squared_error',cv=2,n_jobs=-1)
    ridge_regress.fit(train_dat,train_response)
    ridge = Ridge(ridge_regress.best_params_['alpha'])
    ridge.fit(train_dat,train_response)
    pred_result =ridge.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse
    
    
def lasso_regression(train_dat,train_response,test_dat,test_response):
    lasso_reg = Lasso()
    param = {'alpha':[1e-5,1e-2,1e-1,1,5]}
    lasso_regress = GridSearchCV(lasso_reg,param,scoring='neg_mean_squared_error',cv=2,n_jobs=-1)
    lasso_regress.fit(train_dat,train_response)
    lasso = Lasso(lasso_regress.best_params_['alpha'])
    lasso.fit(train_dat,train_response)
    pred_result =lasso.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse
    
    
def knn_regression(train_dat,train_response,test_dat,test_response):
    knn_reg = KNeighborsRegressor()
    param_grid = {'n_neighbors': np.arange(3, 15)}
    score = 'neg_mean_squared_error'
    knn_gscv = GridSearchCV(knn_reg, param_grid, cv=2, scoring = score,n_jobs=-1)
    knn_gscv.fit(train_dat,train_response)
    knn =  KNeighborsRegressor(knn_gscv.best_params_['n_neighbors'])
    knn.fit(train_dat,train_response)
    pred_result =knn.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse


def neural_nets_regression(train_dat,train_response,test_dat,test_response):
    neur_regr = MLPRegressor(random_state=1)
    scaler = StandardScaler() 
    scaler.fit(train_dat) 
    train_dat = scaler.transform(train_dat)
    test_dat = scaler.transform(test_dat)
    
    tuned_parameters = [{'hidden_layer_sizes': [1,2,3,4,5,6,7,8],
                        'activation': ['relu'],
                        'solver':['lbfgs'], 'alpha':[0.0001],
                        'batch_size':['auto'], 'learning_rate':['constant'],
                        'learning_rate_init':[0.001], 'max_iter':[200]}]
    rgr = GridSearchCV(MLPRegressor(), tuned_parameters,n_jobs=-1, cv=2)
    rgr.fit(train_dat, train_response)
    
    pred_result = rgr.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse
     
    
def svr_regression(train_dat,train_response,test_dat,test_response):
    #model_svr_init = LinearSVR(random_state=2,tol=1e-3)
    #model_svr_init = SVR(kernel = 'rbf')
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
    score = 'neg_mean_squared_error'
    grid = GridSearchCV(SVR(), param_grid, refit = True, scoring = score,n_jobs=-1,cv = 2) 
   
    #model_svr_gs = GridSearchCV(model_svr_init, param_grid, cv=3, scoring = score,n_jobs=24)
    X_scaler = StandardScaler().fit(train_dat)
    X_scaled = X_scaler.transform(train_dat)
    Xtest_norm = X_scaler.transform(test_dat)
    grid.fit(X_scaled,train_response)
    #     model_svr = SVR(model_svr_gs.best_params_['C'])
    #     model_svr.fit(X_scaled,train_response)
    pred_result = grid.predict(Xtest_norm)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse

def random_forest_regression(train_dat,train_response,test_dat,test_response):
    
    RF2 = RandomForestRegressor(random_state=1)
    score = 'neg_mean_squared_error'
    
    param_grid={
                       'max_depth': range(3,7),
                       'n_estimators': (10, 50, 100, 200)
                }

    RF_gscv = GridSearchCV(RF2, param_grid, cv=2, scoring = score,n_jobs=-1)
    RF_gscv.fit(train_dat, train_response)
    RF =  RandomForestRegressor(max_depth= RF_gscv.best_params_['max_depth'],
                                n_estimators= RF_gscv.best_params_['n_estimators'],
                                random_state=1)
   
    RF.fit(train_dat, train_response)
    pred_y = RF.predict(test_dat)
    mse = mean_squared_error(test_response,pred_y)
    RF_rmse = sqrt(mse)
    return RF_rmse

FOLDS = 10
kfold = KFold(FOLDS , shuffle = True, random_state = 123)
rmse_br = np.zeros(FOLDS)
rmse_lr = np.zeros(FOLDS)
rmse_knn = np.zeros(FOLDS)
rmse_svr = np.zeros(FOLDS)
rmse_ridge = np.zeros(FOLDS)
rmse_lasso = np.zeros(FOLDS)
rmse_random_forest = np.zeros(FOLDS)
rmse_neural = np.zeros(FOLDS)
cnt = 0
for train, test in kfold.split(X):  
    train_dat = X[train]
    train_response = y[train]
    test_dat = X[test]
    test_response = y[test]
    rmse_br[cnt] = baseline_regression(train_dat,train_response,test_dat,test_response)
    print ("Baseline done, Linear Regression started!")
    rmse_lr[cnt] = linear_regression(train_dat,train_response,test_dat,test_response)
    print ("Linear Regression done, KNN Regression started!")
    rmse_knn[cnt] = knn_regression(train_dat,train_response,test_dat,test_response)
    print ("KNN Regression done, Svr Regression started!")
    rmse_svr[cnt] = svr_regression(train_dat,train_response,test_dat,test_response)
    print ("SVR Regression done, Ridge Regression started!")
    rmse_ridge[cnt] = ridge_regression(train_dat,train_response,test_dat,test_response)
    print ("Ridge Regression done, Lasso Regression started!")
    rmse_lasso[cnt] = lasso_regression(train_dat,train_response,test_dat,test_response)
    print ("Lasso Regression done, Random Forest started!")
    rmse_random_forest[cnt] = random_forest_regression(train_dat,train_response,test_dat,test_response)
    print ("Random Forest done, Neural nets started!")
    rmse_neural[cnt] = neural_nets_regression(train_dat,train_response,test_dat,test_response)
    print ("Neural nets Regression done!")
    cnt+=1

print ("Results of the 10 Fold CV BR:",rmse_br.mean(),rmse_br.std())
print ("Results of the 10 Fold CV LR:",rmse_lr.mean(),rmse_lr.std())
print ("Results of the 10 Fold CV KNN:",rmse_knn.mean(),rmse_knn.std())
print ("Results of the 10 Fold CV SVR:",rmse_svr.mean(),rmse_svr.std())
print ("Results of the 10 Fold CV Ridge:",rmse_ridge.mean(),rmse_ridge.std())
print ("Results of the 10 Fold CV Lasso:",rmse_lasso.mean(),rmse_lasso.std())
print ("Results of the 10 Fold CV RF:",rmse_random_forest.mean(),rmse_random_forest.std())
print ("Results of the 10 Fold CV Neural_Nets:",rmse_neural.mean(),rmse_neural.std())


