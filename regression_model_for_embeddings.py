##Implement the regression model in the embeddings###
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

def linear_regression(train_dat,train_response,test_dat,test_response):
    LR = LinearRegression(n_jobs = -1)
    LR.fit(train_dat,train_response)
    pred_result = LR.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse

def ridge_regression(train_dat,train_response,test_dat,test_response):
    ridge_reg = Ridge()
    param = {'alpha':[1e-15,1e-10,1e-5,1e-2,1e-1,1,5,10]}
    ridge_regress = GridSearchCV(ridge_reg,param,scoring='neg_mean_squared_error',cv=2,n_jobs = -1)
    ridge_regress.fit(train_dat,train_response)
    ridge = Ridge(ridge_regress.best_params_['alpha'])
    ridge.fit(train_dat,train_response)
    pred_result =ridge.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse
    
    
def lasso_regression(train_dat,train_response,test_dat,test_response):
    lasso_reg = Lasso()
    param = {'alpha':[1e-15,1e-10,1e-5,1e-2,1e-1,1,5,10]}
    lasso_regress = GridSearchCV(lasso_reg,param,scoring='neg_mean_squared_error',cv=2,n_jobs = -1)
    lasso_regress.fit(train_dat,train_response)
    lasso = Lasso(lasso_regress.best_params_['alpha'])
    lasso.fit(train_dat,train_response)
    pred_result =lasso.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse
    
    
def knn_regression(train_dat,train_response,test_dat,test_response):
    knn_reg = KNeighborsRegressor(n_jobs = -1)
    param_grid = {'n_neighbors': np.arange(1, 10)}
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
    neur_regr = MLPRegressor(random_state=1,shuffle=True)
    #scaler = StandardScaler() 
    #scaler.fit(train_dat) 
    #train_dat = scaler.transform(train_dat)
    #test_dat = scaler.transform(test_dat)
    train_dat = preprocessing.scale(train_dat)
    test_dat =  preprocessing.scale(test_dat)
    
    tuned_parameters = [{'hidden_layer_sizes': [2,4,6,8,10,20,30,40],
                       'activation': ['relu'],
                       'solver':['adam'], 'alpha':[0.0001],
                       'batch_size':['auto'], 'learning_rate':['constant'],
                       'learning_rate_init':[0.001], 'max_iter':[500]}]
    #tuned_parameters = [{'hidden_layer_sizes': [2,4,6,8,10,20,30,40],
    #                    'activation': ['relu'],
    #                    'solver':['lbfgs'], 'alpha':[0.0001],
    #                    'batch_size':['auto'], 'learning_rate':['constant'],
    #                    'learning_rate_init':[0.001], 'max_iter':[1200]}]


    

    rgr = GridSearchCV(MLPRegressor(), tuned_parameters, cv=2,n_jobs=-1)
    rgr.fit(train_dat, train_response)
    
    

    #NN.fit(train_dat,train_response)
    #knn =  KNeighborsRegressor(knn_gscv.best_params_['n_neighbors'])
    #neur_regr.fit(train_dat,train_response)
    pred_result = rgr.predict(test_dat)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse
     
    
def svr_regression(train_dat,train_response,test_dat,test_response):
    model_svr_init = LinearSVR(random_state=2,tol=1e-3)
    param_grid = {'C': [0.001,0.01,0.1,1.0, 10]}
    score = 'neg_mean_squared_error'
    model_svr_gs = GridSearchCV(model_svr_init, param_grid, cv=2, scoring = score,n_jobs=-1)
    X_scaler = StandardScaler().fit(train_dat)
    X_scaled = X_scaler.transform(train_dat)
    Xtest_norm = X_scaler.transform(test_dat)
    model_svr_gs.fit(X_scaled,train_response)
    model_svr = LinearSVR(model_svr_gs.best_params_['C'],random_state=2,tol=1e-3)
    model_svr.fit(X_scaled,train_response)
    pred_result = model_svr.predict(Xtest_norm)
    mse = mean_squared_error(test_response,pred_result)
    rmse = sqrt(mse)
    return rmse

def random_forest_regression(train_dat,train_response,test_dat,test_response):
    
    RF2 = RandomForestRegressor(random_state=1,n_jobs=-1)
    score = 'neg_mean_squared_error'
    #     param_grid = {
    #     'max_depth': range(3,15),    
    #     'n_estimators': [50, 150, 250],
    #     'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    #     'min_samples_split': [2, 4, 6]
    #     }
    
    param_grid={
                       'max_depth': range(3,10),
                       'n_estimators': (10, 50, 100, 200)
                }

    RF_gscv = GridSearchCV(RF2, param_grid, cv=2, scoring = score,n_jobs=-1)
    RF_gscv.fit(train_dat, train_response)
    RF =  RandomForestRegressor(max_depth= RF_gscv.best_params_['max_depth'],
                                n_estimators= RF_gscv.best_params_['n_estimators'],
                                random_state=1)
    #     RF =  RandomForestRegressor(max_depth=RF_gscv.best_params_['max_depth'],
    #                                    n_estimators=RF_gscv.best_params_['n_estimators'],
    #                                    min_samples_split = RF_gscv.best_params_['min_samples_split'],
    #                                    random_state=1)

    RF.fit(train_dat, train_response)
    pred_y = RF.predict(test_dat)
    mse = mean_squared_error(test_response,pred_y)
    RF_rmse = sqrt(mse)
    return RF_rmse
data_emb = np.loadtxt("data/RESCAL_EMB_V1.txt")
#data_emb = np.loadtxt("data/COMPLEX_EMB_V1.txt")
#data_emb  = np.loadtxt("data/DISTMUL_EMB_V1.txt")
#data_emb = np.loadtxt("data/TRANSE_EMB_V1.txt")
#data_emb = np.loadtxt("data/HOLE_EMB_V1.txt")
X =  data_emb[:,:-1]
y = data_emb[:,-1]

FOLDS = 10
kfold = KFold(FOLDS , True, 1)
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
    rmse_lr[cnt] = linear_regression(train_dat,train_response,test_dat,test_response)
    print("LR:",rmse_lr[cnt])
    rmse_knn[cnt] = knn_regression(train_dat,train_response,test_dat,test_response)
    print("KNN:",rmse_knn[cnt])
    rmse_svr[cnt] = svr_regression(train_dat,train_response,test_dat,test_response)
    print("SVR:",rmse_svr[cnt])
    rmse_ridge[cnt] = ridge_regression(train_dat,train_response,test_dat,test_response)
    print("RIDGE:",rmse_ridge[cnt])
    rmse_lasso[cnt] = lasso_regression(train_dat,train_response,test_dat,test_response)
    print("LASSO:",rmse_lasso[cnt])
    rmse_random_forest[cnt] = random_forest_regression(train_dat,train_response,test_dat,test_response)
    print("RF:",rmse_random_forest[cnt])
    rmse_neural[cnt] = neural_nets_regression(train_dat,train_response,test_dat,test_response)
    print("ANN:",rmse_neural[cnt])
    cnt+=1

        
print ("Results of the 10 Fold CV LR:",rmse_lr.mean(),rmse_lr.std())
print ("Results of the 10 Fold CV KNN:",rmse_knn.mean(),rmse_knn.std())
print ("Results of the 10 Fold CV SVR:",rmse_svr.mean(),rmse_svr.std())
print ("Results of the 10 Fold CV Ridge:",rmse_ridge.mean(),rmse_ridge.std())
print ("Results of the 10 Fold CV Lasso:",rmse_lasso.mean(),rmse_lasso.std())
print ("Results of the 10 Fold CV RF:",rmse_random_forest.mean(),rmse_random_forest.std())
print ("Results of the 10 Fold CV Neural_Nets:",rmse_neural.mean(),rmse_neural.std())
