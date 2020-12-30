import json
import pandas as pd
import numpy as np
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import stellargraph as sg
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics
import multiprocessing
from stellargraph import datasets
from IPython.display import display, HTML
from stellargraph import datasets, utils
from stellargraph import StellarGraph
from stellargraph import datasets
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from stellargraph.mapper import RelationalFullBatchNodeGenerator
from stellargraph.layer import RGCN
from numpy.random import seed
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import networkx as nx
from nodevectors import Node2Vec
from scipy.sparse import lil_matrix
from rescal import rescal_als


# Establish random seed
RANDOM_SEED = 42

# 1. Set PYTHONHASHSEED environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# 2. Set python built-in pseudo-random generator at a fixed value
import random
random.seed(RANDOM_SEED)

# 3. Set numpy pseudo-random generator at a fixed value
np.random.seed(RANDOM_SEED)

# 4. Set tensorflow pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

def get_embedding_graph_from_tensor(df):
    sub = df["source"].values
    pred = df["orientation"].values
    obj = df["target"].values
    entities = list(set(sub)) + list(set(obj))
    entities.sort()
    relationship = list(set(pred))
    relationship.sort()
    ent = len(entities)
    m = len(relationship)
    X = [lil_matrix((ent,ent)) for i in range(m)]
    
    for index, row in df.iterrows(): 
        X[relationship.index(row["orientation"])][entities.index(row["source"]),entities.index(row["target"])] = 1
    
    A, R, _, _, _ = rescal_als(X, 150, init='random', lambda_A=0.001, lambda_R=0.001)
    
    
    return A,entities

    

####creating graphs#####

df = pd.read_csv('data/patient_regression_analysis_v3.csv',sep=",")
df.head()

patient_ids = df["primaryid"]
drugs = df["drug_name"]
events = df["pt"]
days = df["Days"]
indication = df["indi_pt"]
wt = df["Days"]
gender = df["sex"]
follow_ups = df["i_f_code"]
outcome = df["outc_cod"]
report = df["rept_cod"]
role = df["role_cod"]

edges_with_weights = df[["primaryid","pt","Days"]]



pat_ids = ["pat_"+str(i) for i in range(len(list(set(df["primaryid"].values))))]
adverse_ids = ["adv_"+ str(i) for i in range(len(list(set(df["pt"].values))))]
drug_ids = ["drug_"+ str(i) for i in range(len(list(set(df["drug_name"].values))))]
indi_ids = ["indi_"+ str(i) for i in range(len(list(set(df["indi_pt"].values))))]
gender_ids = ["gender_"+ str(i) for i in range(len(list(set(df["sex"].values))))]
follow_ups_ids = ["follow_"+ str(i) for i in range(len(list(set(df["i_f_code"].values))))]
outcome_ids = ["outcome_"+ str(i) for i in range(len(list(set(df["outc_cod"].values))))]
report_ids = ["report_"+ str(i) for i in range(len(list(set(df["rept_cod"].values))))]
role_ids = ["role_"+ str(i) for i in range(len(list(set(df["role_cod"].values))))]


print ("Total Patient:", len(pat_ids))
print ("Total Adverse Event:", len(adverse_ids))
print ("Total Drug IDs:", len(drug_ids))
print ("Indication IDs:", len(indi_ids))
print ("Gender_ids IDs:", len(gender_ids))
print ("Follow Up IDs:", len(follow_ups_ids))
print ("Outcome IDs:", len(outcome_ids))
print ("Report IDs:", len(report_ids))
print ("Role IDs:", len(role_ids))




##creating a dictionary for patient, adverse event, drugs and indication
pat_ids_dict = {}
patient_info = list(set(df["primaryid"].values))
patient_info.sort()
cnt = 0
for i in patient_info:
    pat_ids_dict[i] = pat_ids[cnt]
    cnt+=1
    
drug_ids_dict = {}
drug_info = list(set(df["drug_name"].values))
drug_info.sort()
cnt = 0
for i in drug_info:
    drug_ids_dict[i] = drug_ids[cnt]
    cnt+=1
    

adverse_ids_dict = {}
adverse_info = list(set(df["pt"].values))
adverse_info.sort()
cnt = 0
for i in adverse_info:
    adverse_ids_dict[i] = adverse_ids[cnt]
    cnt+=1
    

indi_ids_dict = {}
indi_info = list(set(df["indi_pt"].values))
indi_info.sort()
cnt = 0
for i in indi_info:
    indi_ids_dict[i] = indi_ids[cnt]
    cnt+=1
    

gender_ids_dict = {}
gender_info = list(set(df["sex"].values))
gender_info.sort()
cnt = 0
for i in gender_info:
    gender_ids_dict[i] = gender_ids[cnt]
    cnt+=1
    
follow_ups_ids_dict = {}
follow_up_info = list(set(df["i_f_code"].values))
follow_up_info.sort()

cnt = 0
for i in follow_up_info:
    follow_ups_ids_dict[i] = follow_ups_ids[cnt]
    cnt+=1
    
    
outcome_ids_dict = {}
outcome_ids_info = list(set(df["outc_cod"].values))
outcome_ids_info.sort()
cnt = 0
for i in outcome_ids_info:
    outcome_ids_dict[i] = outcome_ids[cnt]
    cnt+=1

report_ids_dict = {}
report_ids_info = list(set(df["rept_cod"].values))
report_ids_info.sort()
cnt = 0
for i in report_ids_info:
    report_ids_dict[i] = report_ids[cnt]
    cnt+=1
    
role_ids_dict = {}
role_ids_info = list(set(df["role_cod"].values))
role_ids_info.sort()
cnt = 0
for i in role_ids_info:
    role_ids_dict[i] = role_ids[cnt]
    cnt+=1

patient_feat = np.identity(len(pat_ids))
adverse_feat = np.identity(len(adverse_ids))
drugs_feat = np.identity(len(drug_ids))
indication_feat = np.identity(len(indi_ids))
gender_feat = np.identity(len(gender_ids))
follow_up_feat =  np.identity(len(follow_ups_ids))
outcome_feat =  np.identity(len(outcome_ids))
report_feat =  np.identity(len(report_ids))
role_feat =  np.identity(len(role_ids))


patient = pd.DataFrame(patient_feat,index= pat_ids)
adverse_effect = pd.DataFrame(adverse_feat,index= adverse_ids)
drugs_presc = pd.DataFrame(drugs_feat,index= drug_ids)
indications =  pd.DataFrame(indication_feat,index=indi_ids)
gender_df =  pd.DataFrame(gender_feat,index=gender_ids)
follow_up_df = pd.DataFrame(follow_up_feat,index=follow_ups_ids)
outcome_feat_df = pd.DataFrame(outcome_feat,index=outcome_ids)
report_feat_df = pd.DataFrame(report_feat,index=report_ids)
role_feat_df = pd.DataFrame(role_feat,index=role_ids)



# # #####Now creating edge information ######

source = []
target = []
orientation = []
days = []
for u,v in zip(patient_ids,drugs):
    source.append(pat_ids_dict[u])
    target.append(drug_ids_dict[v])
    orientation.append("drugs")
    
patient_onset = []
adverse_effect_onset = []
for u,v,w in zip(patient_ids,events,wt):
    source.append(pat_ids_dict[u])
    target.append(adverse_ids_dict[v])
    patient_onset.append(pat_ids_dict[u])
    adverse_effect_onset.append(adverse_ids_dict[v])
    days.append(w)
    orientation.append("adverse_event")
    
for u,v in zip(patient_ids,indication):
    source.append(pat_ids_dict[u])
    target.append(indi_ids_dict[v])
    orientation.append("indication")
    
for u,v in zip(patient_ids,gender):
    source.append(pat_ids_dict[u])
    target.append(gender_ids_dict[v])
    orientation.append("gender")
    
    
for u,v in zip(patient_ids,follow_ups):
    source.append(pat_ids_dict[u])
    target.append(follow_ups_ids_dict[v])
    orientation.append("undergo")
    
for u,v in zip(patient_ids,outcome):
    source.append(pat_ids_dict[u])
    target.append(outcome_ids_dict[v])
    orientation.append("status")
    

for u,v in zip(patient_ids,report):
    source.append(pat_ids_dict[u])
    target.append(report_ids_dict[v])
    orientation.append("submit")
    
for u,v in zip(drugs,role):
    source.append(drug_ids_dict[u])
    target.append(role_ids_dict[v])
    orientation.append("report")
    

for u,v in zip(drugs,indication):
    source.append(drug_ids_dict[u])
    target.append(indi_ids_dict[v])
    orientation.append("purpose")
    
for u,v in zip(drugs,events,):
    source.append(drug_ids_dict[u])
    target.append(adverse_ids_dict[v])
    orientation.append("effect")
    
    

edge_information = pd.DataFrame(
    {"source":source , "target": target,"orientation":orientation}
)

edge_information = edge_information.drop_duplicates()

edges_with_weights = pd.DataFrame({"patient":patient_onset,"adverse_effect":adverse_effect_onset,"wt":days})


print ("***********************")

node_embeddings, node_names = get_embedding_graph_from_tensor(edge_information)


node_names = list(set((list(edge_information["source"].values) + list(edge_information["target"].values))))
node_names.sort()

number_of_embeddings = node_embeddings.shape[1]*2 ##150 features in the model
index = edges_with_weights.index
number_of_rows = len(index)
data_emb = np.zeros((number_of_rows,number_of_embeddings+1))
cnt = 0
for i in range(len(edges_with_weights)): 
    node_pair_1_emb = node_embeddings[node_names.index(edges_with_weights.iloc[i, 0]),]
    node_pair_2_emb = node_embeddings[node_names.index(edges_with_weights.iloc[i, 1]),]
    weight = [edges_with_weights.iloc[i, 2]]
    weight = np.array(weight)
    total_emb_with_weight = np.concatenate((node_pair_1_emb, node_pair_2_emb,weight), axis=None)
    data_emb[cnt,:] =  total_emb_with_weight 
    cnt+=1
 
##save the embeddings from RGCN ####
np.savetxt("data/RESCAL_EMB_only.txt", data_emb)
