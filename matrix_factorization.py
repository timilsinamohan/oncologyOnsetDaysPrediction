import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from sklearn.decomposition import NMF
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import time
import random
from scipy import sparse
from sklearn.model_selection import KFold
import numpy.ma as ma
from scipy.sparse.linalg import svds
import nimfa
import math
from sklearn.preprocessing import StandardScaler

random.seed(123)

def best_component_svd(bg,cmp,train_idx):
    cv_results = {}
    for i in cmp: 
        FOLDS = 3
        IDX = train_idx[:]
        kfold = KFold(FOLDS, True, 1)
        rmse = np.zeros(FOLDS)
        cnt = 0
        true_values = []
        test_pat_id = []
        test_adv_event_id = []
        for idx_train, idx_test in kfold.split(IDX):
            B = bg.copy()
            for j in idx_test:
                test_df = df.iloc[j]
                u = np.where(patient_id==test_df["primaryid"])[0][0]
                v = np.where(adv_event==test_df["pt"])[0][0]
                w = test_df["Days"]
                true_values.append(w)
                test_pat_id.append(u)
                test_adv_event_id.append(v)
                B[u,v]=0
                
            u, s, v = svds(B,k = i)
            reconstructed_matrix = u.dot(np.diag(s).dot(v)) 
            mse = mean_squared_error(true_values,reconstructed_matrix[test_pat_id,test_adv_event_id])
            rmse[cnt] = math.sqrt(mse)
            cnt+=1
        cv_results[i] = rmse.mean()
    best_com = sorted(cv_results, key=cv_results.get, reverse=False)
    return best_com[0]
            


def best_component_pmf(bg,cmp,train_idx):
    cv_results = {}
    for i in cmp: 
        FOLDS = 3
        IDX = train_idx[:]
        kfold = KFold(FOLDS, True, 1)
        rmse = np.zeros(FOLDS)
        cnt = 0
        true_values = []
        test_pat_id = []
        test_adv_event_id = []
        for idx_train, idx_test in kfold.split(IDX):
            B = bg.copy()
            for j in idx_test:
                test_df = df.iloc[j]
                u = np.where(patient_id==test_df["primaryid"])[0][0]
                v = np.where(adv_event==test_df["pt"])[0][0]
                w = test_df["Days"]
                true_values.append(w)
                test_pat_id.append(u)
                test_adv_event_id.append(v)
                B[u,v]=0
                
            
            pmf = nimfa.Pmf(B, seed="random_vcol", rank= i, max_iter=12, rel_error=1e-5)
            pmf_fit = pmf()
            W = pmf_fit.basis()
            H = pmf_fit.coef()
            preds = W.dot(H)
            reconstructed_matrix = preds.A
            mse = mean_squared_error(true_values,reconstructed_matrix[test_pat_id,test_adv_event_id])
            rmse[cnt] = math.sqrt(mse)
            cnt+=1
        cv_results[i] = rmse.mean()
    best_com = sorted(cv_results, key=cv_results.get, reverse=False)
  
    return best_com[0]

            

def best_component_nmf(bg,cmp,train_idx):
    cv_results = {}
    for i in cmp: 
        FOLDS = 3
        IDX = train_idx[:]
        kfold = KFold(FOLDS, True, 1)
        rmse = np.zeros(FOLDS)
        cnt = 0
        true_values = []
        test_pat_id = []
        test_adv_event_id = []
        for idx_train, idx_test in kfold.split(IDX):
            B = bg.copy()
            for j in idx_test:
                test_df = df.iloc[j]
                u = np.where(patient_id==test_df["primaryid"])[0][0]
                v = np.where(adv_event==test_df["pt"])[0][0]
                w = test_df["Days"]
                true_values.append(w)
                test_pat_id.append(u)
                test_adv_event_id.append(v)
                B[u,v]=0
                
            model = NMF(n_components=i, init= 'random')
            WW = model.fit_transform(B)
            HH = model.components_
            WW = sparse.csr_matrix(WW)
            HH = sparse.csr_matrix(HH)
            preds = WW.dot(HH)
            reconstructed_matrix = preds.A
            mse = mean_squared_error(true_values,reconstructed_matrix[test_pat_id,test_adv_event_id])
            rmse[cnt] = math.sqrt(mse)
            cnt+=1
        cv_results[i] = rmse.mean()
    best_com = sorted(cv_results, key=cv_results.get, reverse=False)
    return best_com[0]
            
    

def perform_svd_matrix_reconstruction(B,train_idx):
    cmp = [10,15,20,30,50,70,100]
    get_best_comp = best_component_svd(B,cmp,train_idx)
    #get_best_comp = 20
    u, s, v = svds(B,k = get_best_comp)
    reconstructed_matrix = u.dot(np.diag(s).dot(v)) 
    return reconstructed_matrix




def perform_pmf_matrix_reconstruction(B,train_idx):
    cmp = [10,15,20,30,50,70,100]
    get_best_comp = best_component_pmf(B,cmp,train_idx)
    #get_best_comp = 10
    pmf = nimfa.Pmf(B, seed="random_vcol", rank= get_best_comp, max_iter=12, rel_error=1e-5)
    pmf_fit = pmf()
    W = pmf_fit.basis()
    H = pmf_fit.coef()
    W = sparse.csr_matrix(W)
    H = sparse.csr_matrix(H)
    preds = W.dot(H)
    reconstructed_matrix = preds.A
    return reconstructed_matrix

def perform_nmf_matrix_reconstruction(B,train_idx):
    cmp = [10,15,20,30,50,70,100]
    get_best_comp = best_component_nmf(B,cmp,train_idx)
    #get_best_comp = 20
    model = NMF(n_components=get_best_comp)
    WW = model.fit_transform(B)
    HH = model.components_
    WW = sparse.csr_matrix(WW)
    HH = sparse.csr_matrix(HH)
    preds = WW.dot(HH)
    reconstructed_matrix = preds.A
    return reconstructed_matrix
           
     
def new_innerfold(train_id,test_id):
    ##Ground Truth matrix ##
    B = GT.copy()
    true_values = []
    test_pat_id = []
    test_adv_event_id = []
    for i in test_id:
        test_df = df.iloc[i]
        u = np.where(patient_id==test_df["primaryid"])[0][0]
        v = np.where(adv_event==test_df["pt"])[0][0]
        w = test_df["Days"]
        
        true_values.append(w)
        test_pat_id.append(u)
        test_adv_event_id.append(v)
        
        B[u,v]= np.mean(B[u,:])
    
    
    score = perform_nmf_matrix_reconstruction(B,train_id)
    #score = perform_svd_matrix_reconstruction(B,train_id)
    #score = perform_pmf_matrix_reconstruction(B,train_id)
   
    predicted = score[test_pat_id,test_adv_event_id]
    mse = mean_squared_error(true_values,predicted)
    rmse = math.sqrt(mse)
    print (rmse)
    return rmse
    
    
df = pd.read_csv('data/patient_regression_analysis_v3.csv',sep=",")
df = df[["primaryid","pt","Days"]]

m = len(df["primaryid"].unique())
n = len(df["pt"].unique())

IDX = list(range(df.shape[0]))
 


FOLDS = 10
GT = sparse.lil_matrix((m,n))
#GT_norm = sparse.lil_matrix((m,n))
print (GT.shape)
patient_id = df["primaryid"].unique()
patient_id.sort()
adv_event = df["pt"].unique()
adv_event.sort()
###prepare the matrix#####
for index, row in df.iterrows():
    GT[np.where(patient_id==row["primaryid"])[0][0], np.where(adv_event==row["pt"])[0][0]]= row["Days"]


RMSE = np.zeros(FOLDS)
kfold = KFold(FOLDS, True, 1)
cnt = 0
for idx_train, idx_test in kfold.split(IDX):  
    start_time = time.time()
    RMSE[cnt]= new_innerfold(idx_train,idx_test)
    print("--- %s seconds ---" % (time.time() - start_time)) 
    cnt+=1


print ("Mean RMSE", RMSE.mean()," ", "Standard Deviation:", RMSE.std())

