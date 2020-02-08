"""
This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from gensim.summarization.textcleaner import split_sentences
import scipy.spatial
import numpy as np
import datetime
import os.path

def cluster_semsim(embed, embed_test):
    dists = scipy.spatial.distance.cdist(embed, [embed_test], "cosine")
    avg_dist = np.mean(dists)
    #print(avg_dist)
    return 1 - avg_dist 

# Evaluate soft-cosine distance to enhance clusters cohesion
def cluster_semsim_multi_multi(embed, embed_test):
    dists = scipy.spatial.distance.cdist(embed, embed_test, "cosine")
    #print(dists.shape)
    avg_dist = np.mean(dists)
    print("avg dist shape:: {}".format(avg_dist.shape))
    return 1 - avg_dist 

print("start: tsv file import {}".format(datetime.datetime.now().time()))
#data = np.genfromtxt("corpus/tech_auto.tsv", delimiter="\t", dtype=None, encoding=None)
data = np.genfromtxt("corpus/legal.tsv", delimiter="\t", dtype=None, encoding=None)
l_sentences = []
for array in data:
    sents = split_sentences(array[4])
    for sent in sents:
        l_sentences.append(sent)
print("stop: tsv file import {}".format(datetime.datetime.now().time()))
print("")

#embedder = SentenceTransformer('bert-base-nli-mean-tokens')
print("start: embedder init {}".format(datetime.datetime.now().time()))
#embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
print("stop: embedder init {}".format(datetime.datetime.now().time()))

# Test Corpus
#test_sentences = ['automotive','sport','abhjkefiu','legal','agricolture']
#corpus_test_embeddings = embedder.encode(test_sentences)

#check id embedding exists
if os.path.isfile('corpus_test_embeddings_base_legal.mbd.npy'):
    print ("File embeddings exist")
    corpus_test_embeddings = np.load('corpus_test_embeddings_base_legal.mbd.npy') 
    #print("test em eddings: {}".format(corpus_test_embeddings))

else:
    print ("File embeddings not exist")
    print("start: corpus test embedding {}".format(datetime.datetime.now().time()))
    corpus_test_embeddings = embedder.encode(l_sentences)
    print("stop: corpus test embedding {}".format(datetime.datetime.now().time()))
    np.save('corpus_test_embeddings_base_legal.mbd', corpus_test_embeddings)

# Main corpus
f = open("test.txt","r")
contents = f.read()
corpus = split_sentences(contents)
corpus_embeddings = embedder.encode(corpus)

# Perform kmean clustering
num_clusters = 4
clustering_model = KMeans(n_clusters=num_clusters, random_state=1115)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
clustered_embeddings = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])
    clustered_embeddings[cluster_id].append(corpus_embeddings[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")

# Try to identify a cluster distance from topic
print("\t\tTesla\tSport\tRandom\tLegal\tAgricolture")
for c in range(num_clusters):
    print("Cluster {}  ".format(c+1),end='')
    #for corpus_test_embedding in corpus_test_embeddings:
    ss = cluster_semsim_multi_multi(clustered_embeddings[c], corpus_test_embeddings)
    print("\t{}".format(ss), end='')
    print("")
    
#for s,test_sent in enumerate(test_sentences):
#    print("###########################################################################################")
#    print("Test sentence #{}: {} ".format(s, test_sent))
#    distances = scipy.spatial.distance.cdist([corpus_test_embeddings[s]], corpus_embeddings, "cosine")[0]
#
#    avg_similarity = 0
#    for i, distance in enumerate(distances):
#         #print(corpus[i])
#         #print("# Similarity = {} ".format(distance))
#         avg_similarity += distance
#         #print("======================================================================================")
#    avg_similarity = 1 - (avg_similarity / len(distances))
#    print("Avg similarity of sentence: = {}".format(avg_similarity))
#    print("##########################################################################################")

