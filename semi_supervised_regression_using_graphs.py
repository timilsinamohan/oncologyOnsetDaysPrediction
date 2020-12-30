import pandas as pd
from sklearn import preprocessing
import networkx as nx 
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import lil_matrix
from scipy import linalg
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import math
import warnings
warnings.filterwarnings("ignore")
from scipy import sparse


def get_baseline_prediction(trained_labels_values,test_nodes):
    pred = np.mean(trained_labels_values)
    state = np.ones(len(trained_labels_values)+ len(test_nodes))*pred
    return state

def get_personalized_pagerank_scores(G,train_node_id,test_node_id,alpha):
    n = G.shape[0]
    graph = G.copy()
    H = normalize(graph.T, norm='l1', axis=0)
    all_values = get_true_value.copy()
    all_values[test_node_id] = 0
    state_vector = all_values.copy() 
    state = state_vector.copy()
    g = (1.0/n * np.ones(n))
    #g = g.reshape(n,1)
    state = state.reshape(n,1)
    R = alpha * H + (1.00-alpha) * g
       
    ###Propagate the scores####
    
    iteration = 100
    for j in range(iteration):
        state_matrix_next = R.dot(state)
        state = state_matrix_next.copy()
  
    
    return state

def get_katz_scores(G,train_node_id,test_node_id,alpha):
    n = G.shape[0]
    graph = G.copy()
    all_values = get_true_value.copy()
    all_values[test_node_id] = 0
    state_vector = all_values.copy() 
    state = state_vector.copy()
       
    ###Propagate the scores####
    
    iteration = 100
    beta = 100.0
    for j in range(iteration):
        state_matrix_next = G.dot(alpha * state)+ beta
        state = state_matrix_next.copy()
  
    
    return state

def get_lgc_scores(G,train_node_id,test_node_id,alpha):
     
    """ LGC computes the normalized Laplacian as its propagation matrix"""
    n = G.shape[0]
    graph = G.copy()  
    degrees = graph.sum(axis=0).A[0]
    degrees[degrees==0] += 1  # Avoid division by 0
    D2 = np.sqrt(sparse.diags((1.0/degrees),offsets=0))
    S = D2.dot(graph).dot(D2)
    S = S*alpha
    
    ###create Base matrix#######
    
    all_values = get_true_value.copy()
    all_values[test_node_id] = 0
    state_vector = all_values.copy() 
    
    
    ###Propagate the scores####
    remaining_iter = 100
    state = state_vector.copy()
    
    Base = state_vector
    while remaining_iter > 0:
        state = S.dot(state) + Base*(1-alpha)
        remaining_iter -= 1
    
    state = np.round_(state, decimals=2)
    return state

def get_hd_scores(G,train_node_id,test_node_id,alpha):
    n = G.shape[0]
    graph = G.copy()
    all_values = get_true_value.copy()
    all_values[test_node_id] = 0
    state_vector = all_values.copy() 
       
    ###Propagate the scores####
    remaining_iter = 30
    state = state_vector.copy()
    state = state.reshape(n,1)
    I = np.eye(n,n,dtype=np.float64)
    L = sparse.csgraph.laplacian(graph,normed=True) 
    V = I + (-alpha/remaining_iter) * L
        
    while remaining_iter > 0:
        state = V.dot(state)
        remaining_iter -= 1
    
    return state


def get_harmonic_scores(G,train_node_id,test_node_id,yl):
    
    ##create Propagation Matrix###

    n = G.shape[0]
    graph = G.copy()
    degrees = graph.sum(axis=0).A[0]
    degrees[degrees==0] += 1  # Avoid division by 0
    D = sparse.diags((1.0/degrees),offsets=0)
    P = D.dot(graph).tolil()
    P[train_node_id] = 0
    all_values = np.zeros(n)
    all_values[train_node_id] = yl

    all_values[test_node_id] = 0
    state_vector = all_values.copy()
    
    ###Propagate the scores####
    remaining_iter = 30
    state = state_vector.copy()
    Base = state.copy()
    P = P.A
    
    while remaining_iter > 0:
        state = P.dot(state) + Base
        remaining_iter -= 1
        
    state = np.round_(state, decimals=2)

   
    return state

def get_knn_graph(X,training_nodes):
    knn_reg = KNeighborsRegressor()
    param_grid = {'n_neighbors': [2,5,7,10,20]}
    score = 'neg_mean_squared_error'
    knn_gscv = GridSearchCV(knn_reg, param_grid, cv=3, scoring = score,n_jobs=24)
    train_dat = X[training_nodes]
    knn_gscv.fit(train_dat,y[training_nodes])
    print ("best number of neighbor is:",knn_gscv.best_params_['n_neighbors'])
    GF = kneighbors_graph(X, knn_gscv.best_params_['n_neighbors'], mode='connectivity', include_self=False)
    return GF
 
def get_graph_matrix(X,training_nodes):
  
    GF = get_knn_graph(X,training_nodes)
    
    return GF

def innerfold(test_nodes, train_nodes):
    true_values = get_true_value.copy()
    graph_data = get_graph_matrix(data_feature,train_nodes)
    trained_labels_values = true_values[train_nodes]
    #predicted_values = get_baseline_prediction(trained_labels_values,test_nodes)
    predicted_values = get_harmonic_scores(graph_data,train_nodes,test_nodes,trained_labels_values)
    #predicted_values = get_hd_scores(graph_data,train_nodes,test_nodes,alpha = 1.0)
    #predicted_values = get_bhd_scores(graph_data,train_nodes,test_nodes,alpha = 1.0)
    #predicted_values = get_lgc_scores(graph_data,train_nodes,test_nodes,alpha = 0.5)
    #predicted_values = get_katz_scores(graph_data,train_nodes,test_nodes,alpha = 0.001)
    predicted_values = get_personalized_pagerank_scores(graph_data,train_nodes,test_nodes,alpha = 0.85)
    mse = mean_squared_error(true_values[test_nodes],predicted_values[test_nodes])
    rmse = math.sqrt(mse)
    print (rmse)
    #print accuracy
    return rmse


if __name__ == '__main__':
    df = pd.read_csv('data/patient_regression_analysis_v3.csv',sep=",")
    df["primaryid"] = df["primaryid"].astype(str)
    #convert the categorical columns into numeric
    le = preprocessing.LabelEncoder()
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
    data_feature = X.copy()
    n = data_feature.shape[0]
    get_true_value = y.copy()
    x = np.arange(n) 
    FOLDS = 10
    cnt = 0
    kfold = KFold(FOLDS, True, random_state=123)
    rmse_test = np.zeros(FOLDS)
    for test_nodes_index, train_nodes_index in kfold.split(X):  
        test_nodes = test_nodes_index
        train_nodes = train_nodes_index
        rmse_test[cnt] = innerfold(test_nodes, train_nodes)
        cnt+=1
        
    
    print('RMSE Test Mean / Std: %f / %f' % (rmse_test.mean(), rmse_test.std()))



