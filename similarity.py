from sentence_clustering import SentenceSimilarity
import argparse

# available models:
# 'bert-base-nli-mean-tokens'
# 'roberta-large-nli-stsb-mean-tokens'
# 'roberta-base-nli-stsb-mean-tokens'

def run():
    parser = argparse.ArgumentParser(description='Process input text and produces topic similarity scores')
    parser.add_argument('-path', dest='path', default=None, help='File path of sample text')
    parser.add_argument('-model', dest='model', default='bert-large-uncased', help='')
    parser.add_argument('-dist-algo', dest='dist_algo', default='cosine', help='Which algorythm use to compute distance between N-dimensional embeddings')
    parser.add_argument('-topic', dest='topic', default='legal', help='Topic to compute similarity with text in input')
    args = parser.parse_args()

    if not args.path:
        raise RuntimeError("Must supply text path.")

    model = SentenceSimilarity(
        model=args.model,
        dist_algo=args.dist_algo,
        topic=args.topic,
	path=args.path 
    )

    model.run()

if __name__ == '__main__':
    run()

