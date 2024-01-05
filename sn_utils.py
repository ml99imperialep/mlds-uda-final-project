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

def k_means_clustering(k, embeddings):
    """
    Define K Means clustering with number of klusters k on the embeddings
    """
    num_clusters = k 
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels


def sent_relation(relationship_df):
    """
    Derive sentiments relation with regards of the sentiments of the character:
    If two characters are positive, their relation is positive.
    If two characters are negative, their relation is negative.

    If their sentiment is different the sentiment of their relation is based on the number of interactions "value", if it is bigger than the mean value and the source is positive, the relation is positive.
    If it is the way around then it is negative. If the value is smaller than the mean value it is again the same around and the sentiment of the target the sentiment of the relation.
    """
    sent_relation = []
    for i in range(len(relationship_df)):
    
        if relationship_df['sentiments_source'][i] == [1]:
            if relationship_df['sentiments_target'][i] == [1]:
                sent_relation.append(1)
            else:
                if relationship_df['value'][i] > relationship_df['value'].mean():
                   sent_relation.append(0)
                else:
                    sent_relation.append(1)
        else:
            if relationship_df['sentiments_target'][i] == [0]:
                sent_relation.append(0)
            else:
                if relationship_df['value'][i] > relationship_df['value'].mean():
                    sent_relation.append(1)
                else:
                    sent_relation.append(0)
    return sent_relation


