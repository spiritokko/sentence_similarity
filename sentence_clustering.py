from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from gensim.summarization.textcleaner import split_sentences
import scipy.spatial
import numpy as np
import datetime
import os.path

class SentenceSimilarity ():
   def __init__(self, model, dist_algo, topic, path):
      self.model = model
      self.dist_algo = dist_algo
      self.topic = topic
      self.path = path
   def run(self):
      print("Starting sentence similarity model with params: {}, {}, {}, {}".format(self.model, self.dist_algo, self.topic, self.path))
      execute(self.model, self.dist_algo, self.topic, self.path)
      print("Exiting")

def cluster_semsim(embed, embed_test, dist_algo):
    dists = scipy.spatial.distance.cdist(embed, [embed_test], dist_algo)
    avg_dist = np.mean(dists)
    return 1 - avg_dist 

def cluster_semsim_multi_multi(embed, embed_test, dist_algo):
    dists = scipy.spatial.distance.cdist(embed, embed_test, dist_algo)
    avg_dist = np.mean(dists)
    return 1 - avg_dist 

#topic = tech_auto, legal
def execute(model, dist_algo, topic, path):
#1) Load model
		
	print("start: embedder init {}".format(datetime.datetime.now().time()))
	embedder = SentenceTransformer(model)
	print("stop: embedder init {}".format(datetime.datetime.now().time()))

#2) Check if embeddings for topic exists else read tsv topic file and create embeddings
	str_topic_embedding_file = 'sc_' + model + '_' + topic + '.mbd.npy' 
	if os.path.isfile(str_topic_embedding_file):
		print ("File embeddings exist")
		corpus_test_embeddings = np.load(str_topic_embedding_file)
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


#3) Load sample file
	f = open(path,"r")
	contents = f.read()
	corpus = split_sentences(contents)
	corpus_embeddings = embedder.encode(corpus)


#4) Perform kmean clustering
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

#5) Compute cluster distance from topic
	for c in range(num_clusters):
    		print("Cluster {}  ".format(c+1),end='')
    		ss = cluster_semsim_multi_multi(clustered_embeddings[c], corpus_test_embeddings, dist_algo)
    		print("\t{}".format(ss), end='')
    		print("")

# Sample topics
#test_sentences = ['automotive','sport','abhjkefiu','legal','agricolture']
#corpus_test_embeddings = embedder.encode(test_sentences)
