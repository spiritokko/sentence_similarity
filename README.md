# Sentence Clustering and Similarity
This is a sample application to extract semantic information from raw text and derive _topic_ similarity.
It is based on UKPLab Sentence Transformers code (https://github.com/UKPLab/sentence-transformers).
The architecture is as follows:
1) The raw text passed in input (there are already some sample texts in the directory _samples_) is converted in sentence embeddings using SentenceTransformers model
2) Sentence embeddings for specific topic is also loaded from the filesystem, if available, or computed on the fly: the source data for each topic is derived from a separate application that uses Wikipedia APIs. WE chose the 'summary' API instead of the full article, because it showed much better results.
3) The sentences in input text are then clustered with KMeans algorythm, passing embeddings for each sentence.
4) Each cluster of sentences is compared semantically to each topic's embeddings and a similarity score for each cluster is calculated.

It's worth noting that new topics can be added to extend the domain of comparison, as long as they respect a custom tsv format

## Sample command line usage

```
python similarity.py -nclusters 4 -path samples/electric_pickup_war.txt -model bert-base-wikipedia-sections-mean-tokens -topic all
```
