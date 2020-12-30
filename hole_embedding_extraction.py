import pandas as pd
from ampligraph.latent_features import ComplEx
from ampligraph.latent_features import ConvE
from ampligraph.latent_features import ConvKB
from ampligraph.latent_features import HolE
from ampligraph.latent_features import DistMult
from ampligraph.latent_features import TransE


import numpy as np

#df = pd.read_csv('relational_graph_data.txt',sep=",")
df = pd.read_csv('data/relational_graph_data_v1.txt',sep = ",")
df = df[['source', 'orientation', 'target']]
X = df.to_numpy()
print ("started to create the model")


model = HolE(batches_count=1, seed=555, epochs=100, k=75, eta=5,
                 loss='pairwise', loss_params={'margin':1},
                regularizer='LP', regularizer_params={'lambda':0.1})



print ("Model fitting started")
model.fit(X)
print ("Model fitting completed")
nds = list(set((list(df["source"].values) + list(df["target"].values))))
nds.sort()
print ("Embedding model started")
emb = model.get_embeddings(nds,embedding_type='entity')
print ("shape of the embedding vector:",emb.shape)
print ("Embedding model completed")
number_of_embeddings = emb.shape[1]*2 ##150 features in the model
index = df.index
number_of_rows = len(index)
data_emb = np.zeros((number_of_rows,number_of_embeddings+1))
cnt = 0
edges_with_weights = pd.read_csv('data/edges_with_weights.txt',sep=",")

for i in range(len(edges_with_weights)): 
    node_pair_1_emb = emb[nds.index(edges_with_weights.iloc[i, 0]),]
    node_pair_2_emb = emb[nds.index(edges_with_weights.iloc[i, 1]),]
    weight = [edges_with_weights.iloc[i, 2]]
    weight = np.array(weight)
    total_emb_with_weight = np.concatenate((node_pair_1_emb, node_pair_2_emb,weight), axis=None)
    data_emb[cnt,:] =  total_emb_with_weight 
    cnt+=1
    
   
np.savetxt("data/COMPLEX_EMB_V1.txt", data_emb)

