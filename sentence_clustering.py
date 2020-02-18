from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization.textcleaner import split_sentences
from typing import List
import numpy as np
import datetime
import os.path

class SentenceSimilarity ():
   def __init__(self, model, dist_algo, topic, path, nclusters):
      self.model = model
      self.dist_algo = dist_algo
      self.topic = topic
      self.path = path
      self.nclusters = nclusters
   def run(self):
      print("Starting sentence similarity model with params: {}, {}, {}, {}, {}".format(self.model, self.dist_algo, self.topic, self.path, self.nclusters))
      execute(self.model, self.dist_algo, self.topic, self.path, self.nclusters)
      print("Exiting")


def cluster_semsim(embed, embed_test, dist_algo):
    dists = scipy.spatial.distance.cdist(embed, [embed_test], dist_algo)
    avg_dist = np.mean(dists)
    return 1 - avg_dist 


def cluster_semsim_multi_multi(embed, embed_test, dist_algo):
    cosine_scores = 1 - (paired_cosine_distances(embed, embed_test))
    cosine_scores_mean = np.mean(cosine_scores)
    return cosine_scores_mean 


def cluster_cosine_similarity(embed, embed_test):
    cosine_scores = cosine_similarity(embed, embed_test)
    return np.mean(cosine_scores)


def list_topics() -> List[str]:
	f = open('corpus/topics.txt',"r")
	topics = f.readlines()
	return topics

def cluster_this(n_clusters, corpus_embeddings, n_init, max_iter, tol):
    print("KMeans params: {} {} {} ".format(n_init, max_iter, tol))
    clustering_model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_centers = clustering_model.cluster_centers_
    #print(cluster_centers)
    return cluster_assignment


def load_or_create_topic_embedding(model, topic, embedder):
	str_topic_embedding_file = 'sc_' + model + '_' + topic + '.mbd.npy' 
	
	# TODO: It is not enough to check if binary for embeddings already exists: we have also to check if the original raw data file has changed e.g. new data included
	if os.path.isfile(str_topic_embedding_file):
		print ("File embeddings exist")
		corpus_test_embeddings = np.load(str_topic_embedding_file, allow_pickle=True)
	else:
		print ("File embeddings not exist")

		print("start: tsv file import {}".format(datetime.datetime.now().time()))
		data = np.genfromtxt("corpus/"+topic+".tsv", delimiter="\t", dtype=None, encoding=None)
		l_sentences = []
		for array in data:
			sents = split_sentences(array[4])
			for sent in sents:
				l_sentences.append(sent)
		print("stop: tsv file import {}".format(datetime.datetime.now().time()))
		print("")

		print("start: corpus test embedding {}".format(datetime.datetime.now().time()))
		corpus_test_embeddings = embedder.encode(l_sentences)
		print("stop: corpus test embedding {}".format(datetime.datetime.now().time()))
		np.save('sc_' + model + '_' + topic + '.mbd', corpus_test_embeddings)
	return corpus_test_embeddings


def execute(model, dist_algo, topic, path, nclusters):
#1) Load model
		
	print("start: embedder init {}".format(datetime.datetime.now().time()))
	embedder = SentenceTransformer(model)
	print("stop: embedder init {}".format(datetime.datetime.now().time()))

#2) Topic management: if topic parameter is a valid topic argument it is load from disk or its embedding computed. If it is all, every topic available is managed
#   Check if embeddings for topic exists else read tsv topic file and create embeddings
	topic_embeddings = {} 
	if topic == 'all':
		topic_embeddings["Legal_terminology"] = load_or_create_topic_embedding(model, 'Legal_terminology', embedder) # Legal
		topic_embeddings["Automotive_technologies"] = load_or_create_topic_embedding(model, 'Automotive_technologies', embedder) # Automotive
		topic_embeddings["Investment"] = load_or_create_topic_embedding(model, 'Investment', embedder) # Finance
	else:
		topic_embeddings[topic] = load_or_create_topic_embedding(model, topic, embedder)

#3) Load sample file
	f = open(path,"r")
	contents = f.read()
	corpus = split_sentences(contents)
	corpus_embeddings = embedder.encode(corpus)


#4) Perform kmean clustering
	num_clusters = int(nclusters)
	print()
	print("@@@@@@@@@@@@@@@@@@@@@@ iter #{}: @@@@@@@@@@@@@@@@@@@@@@".format(iter))
	cluster_assignment = cluster_this(num_clusters, corpus_embeddings, 50, 500, 1e-6)
	clustered_sentences  = [[] for i in range(num_clusters)]
	clustered_embeddings = [[] for i in range(num_clusters)]
	for sentence_id, cluster_id in enumerate(cluster_assignment):
    		clustered_sentences[cluster_id].append(corpus[sentence_id])
    		clustered_embeddings[cluster_id].append(corpus_embeddings[sentence_id])

	for i, cluster in enumerate(clustered_sentences):
		tpc_score = []
		for tpc,emd in topic_embeddings.items():
			ss = cluster_cosine_similarity(clustered_embeddings[i], emd)
			tpc_score.append(ss)
			print("Cluster #{}: Topic {}: Similarity {}: ".format(i+1, tpc, ss))
		print("Cluster #{}: Reference Topic {}: Variance {}: ".format(i+1, np.amax(tpc_score), np.var(tpc_score)))
		#print(cluster)
		print("****************************************************************************")

