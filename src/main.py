#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse, logging
import numpy as np
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
from log import set_up_log
from data import *
from train import *
from model import *
import graph
import random
import os


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
	'''
	Parses the struc2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run struc2vec.")
	parser.add_argument('--dataset', type=str, default='celegans', help='dataset name')
	parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
	parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of the test against whole')
	parser.add_argument('--epoch', type=int, default=3000, help='number of epochs')
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--bs', type=int, default=128, help='minibatch size')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--l2', type=float, default=0, help='l2 regularization weight')
	parser.add_argument('--use_cache', action='store_true', help='whether to use cached embeddings')
	parser.add_argument('--workers', type=int, default=30,
	                    help='Number of parallel workers. Default is 8.')
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=None,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')



	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--OPT1', default=False, type=bool,
                      help='optimization 1')
	parser.add_argument('--OPT2', default=False, type=bool,
                      help='optimization 2')
	parser.add_argument('--OPT3', default=False, type=bool,
                      help='optimization 3')

	parser.add_argument('--log_dir', type=str, default='./log/', help='root directory for storing logs')  # sp (shortest path) or rw (random walk)
	args = parser.parse_args()
	return args

def read_graph(args):
	'''
	Reads the input network.
	'''
	logging.info(" - Loading graph...")
	G = graph.load_edgelist(args.input,undirected=True)
	logging.info(" - Graph loaded.")
	return G

def learn_embeddings():
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	walks = LineSentence('random_walks.txt')
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	logging.info("Representations created.")

	return

def exec_struc2vec(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	if(args.OPT3):
		until_layer = args.until_layer
	else:
		until_layer = None

	G = read_graph(args)
	G = struc2vec.Graph(G, args.directed, args.workers, untilLayer=until_layer)

	if(args.OPT1):
		G.preprocess_neighbors_with_bfs_compact()
	else:
		G.preprocess_neighbors_with_bfs()

	if(args.OPT2):
		G.create_vectors()
		G.calc_distances(compactDegree = args.OPT1)
	else:
		G.calc_distances_all_vertices(compactDegree = args.OPT1)


	G.create_distances_network()
	G.preprocess_parameters_random_walk()

	G.simulate_walks(args.num_walks, args.walk_length)


	return G


def optimize_struc2vec_embeddings(args):
	G = exec_struc2vec(args)
	learn_embeddings()


def main(args):
	set_up_log(args)
	set_random_seed(args)
	_, labels, set_indices, train_mask, test_mask = get_data(args) # get train/test mask, set indices; store the processed graph as .edgelist
	if not args.use_cache:
		optimize_struc2vec_embeddings(args) # pack original code
	set_random_seed(args)
	features = load_features(args, set_indices)  # shape: [N, set_size, F]
	model = create_model(features, labels)
	best_test_acc = train(model, features, labels, train_mask, test_mask, args)


if __name__ == "__main__":
	args = parse_args()
	main(args)

