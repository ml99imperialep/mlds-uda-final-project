#################################
# This file contains all functions which are used in the Knowledge Graph part of social_network.ipynb
#################################

from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np

import pandas as pd


#### It  must be notes that this function is copied from https://github.com/thu-vu92/the_witcher_network and from https://github.com/ohumu/HarryPotterNetwork/blob/main/extracting_relationships.ipynb to compare this model with the sentiment social network
def filter_entity(ent_list, character_df, name_mapping):
    """
    Filtering the entities which are in character_df and analyse if they are in the variation of the character_df
    """
    character_names = character_df['Name'].tolist()
    variation = character_df['Variation'].tolist()
    processed_names = []
    # analyse if the name is in the Name column of character_df or not
    for name, variation in zip(character_names, variation):
        processed_name = name_mapping.get(name, name)
        processed_variation = []
        # if it has no variation then 
        if pd.notna(variation):
            processed_variation = [name_mapping.get(var.strip(), var.strip()) for var in variation.split(';')]

        processed_names.append(processed_name)
        processed_names.extend(processed_variation)

    return [ent for ent in ent_list if ent in processed_names]

# functions for Graph Embedding

def word_embedding(G, dim, walk_length, num_walks, workers):
    """
    Define word embedding with Node2Vec and a training model with batch words of 4 and a window of 10
    """
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers) 
    model = node2vec.fit(window=10, min_count=1, batch_words=4) 
    return model

def embedding(model, G, perplexity, n_iter):
    """
    Define t-sne embedding on the trained model of word_embedding and G with perplexity and n_iter to be chosen
    """
    # Get embeddings for all nodes
    embeddings = np.array([model.wv[node] for node in G.nodes()])

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

# function for Clustering 

def k_means_clustering(k, embeddings):
    """
    Define K Means clustering with number of klusters k on the embeddings
    """
    num_clusters = k 
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels



