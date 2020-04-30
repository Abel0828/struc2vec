import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import random
from itertools import combinations
import logging

def get_data(args):
    (G, labels), task = read_file(args)
    if labels is None:
        logging.info('Labels unavailable. Generating training/test instances from dataset ...')
        G, labels, set_indices, (train_mask, test_mask) = generate_set_indices_labels(G, task, test_ratio=args.test_ratio)
        logging.info(' Generate {} pos/neg training & test instances in total.'.format(set_indices.shape[0]))
    else:
        # training on nodes or running on synthetic data
        logging.info('Labels available (node-level task) or no need to train model')
        set_indices = np.expand_dims(np.arange(G.number_of_nodes()), 1)
        train_mask, test_mask = split_dataset(set_indices.shape[0], test_ratio=args.test_ratio)
    logging.info('Training size {}:, test size: {}, test ratio: {}'.format(int(train_mask.sum()), int(test_mask.sum()), args.test_ratio))
    nx.write_edgelist(G, 'graph/{}{}.edgelist'.format(args.dataset, args.seed))
    args.dataset = args.dataset + str(args.seed)

    return G, labels, set_indices, train_mask, test_mask


def read_file(args):
    dataset = args.dataset
    di_flag = args.directed
    if dataset in ['brazil-airports', 'europe-airports', 'usa-airports', 'foodweb']:
        task = 'node_classification'
    elif dataset in ['arxiv', 'celegans', 'celegans_small', 'facebook', 'ns', 'pb', 'power', 'router', 'usair', 'yeast', 'grid']:
        task = 'link_prediction'
    elif dataset in ['arxiv_tri', 'celegans_tri', 'celegans_small_tri', 'facebook_tri', 'ns_tri', 'pb_tri',
                     'power_tri', 'router_tri', 'usair_tri', 'yeast_tri', 'triangular_grid_tri', 'foodweb_tri']:
        task = 'triplet_prediction'
    else:
        raise ValueError('dataset not found')

    directory = './data/' + task + '/' + dataset + '/'

    labels, node_id_mapping = read_label(directory, task=task)
    edges = read_edges(directory, node_id_mapping)

    if not di_flag:
        G = nx.Graph(edges)
    else:
        G = nx.DiGraph(edges)
    logging.info('Read in {} for {} --  number of nodes: {}, number of edges: {}, number of labels: {}. Directed: {}'.format(dataset, task,
                                                                                                                G.number_of_nodes(),
                                                                                                                G.number_of_edges(),
                                                                                                                len(labels) if labels is not None else 0,
                                                                                                               di_flag))
    labels = np.array(labels) if labels is not None else None
    return (G, labels), task


def read_label(dir, task):
    if task == 'node_classification':
        f_path = dir + 'labels.txt'
        fin_labels = open(f_path)
        labels = []
        node_id_mapping = dict()
        for new_id, line in enumerate(fin_labels.readlines()):
            old_id, label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = new_id
        fin_labels.close()
    else:
        labels = None
        nodes = []
        with open(dir + 'edges.txt') as ef:
            for line in ef.readlines():
                nodes.extend(line.strip().split()[:2])
        nodes = sorted(list(set(nodes)))
        node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return labels, node_id_mapping


def read_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def generate_set_indices_labels(G, task, test_ratio):
    G = G.to_undirected()  # the prediction task completely ignores directions
    pos_edges, neg_edges = sample_pos_neg_sets(G, task)  # each shape [n_pos_samples, set_size], note hereafter each "edge" may contain more than 2 nodes
    n_pos_edges = pos_edges.shape[0]
    assert(n_pos_edges == neg_edges.shape[0])
    pos_test_size = int(test_ratio * n_pos_edges)

    set_indices = np.concatenate([pos_edges, neg_edges], axis=0)
    test_pos_indices = random.sample(range(n_pos_edges), pos_test_size)  # randomly pick pos edges for test
    test_neg_indices = list(range(n_pos_edges, n_pos_edges + pos_test_size))  # pick first pos_test_size neg edges for test
    test_mask = get_mask(test_pos_indices + test_neg_indices, length=2*n_pos_edges)
    train_mask = np.ones_like(test_mask) - test_mask
    labels = np.concatenate([np.ones((n_pos_edges, )), np.zeros((n_pos_edges, ))]).astype(np.int32)
    G.remove_edges_from([node_pair for set_index in list(set_indices[test_pos_indices]) for node_pair in combinations(set_index, 2)])

    # permute everything for stable training
    permutation = np.random.permutation(2*n_pos_edges)
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    train_mask, test_mask = train_mask[permutation], test_mask[permutation]

    return G, labels, set_indices, (train_mask, test_mask)


def sample_pos_neg_sets(G, task):
    if task == 'link_prediction':
        pos_edges = np.array(list(G.edges), dtype=np.int32)
        neg_edges = np.array(sample_neg_sets(G, pos_edges.shape[0], set_size=2), dtype=np.int32)
    elif task == 'triplet_prediction':
        pos_edges = np.array(collect_tri_sets(G))
        neg_edges = np.array(sample_neg_sets(G, pos_edges.shape[0], set_size=3), dtype=np.int32)
    else:
        raise NotImplementedError

    return pos_edges, neg_edges


def collect_tri_sets(G):
    tri_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def sample_neg_sets(G, n_samples, set_size):
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        for node1, node2 in combinations(candid_set, 2):
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break

    return neg_sets


def split_dataset(n_samples, test_ratio=0.2):
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio)
    train_mask = get_mask(train_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, test_mask


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_features(args, set_indices):
    dataset = args.dataset
    f_path = 'emb/{}.emb'.format(dataset)
    embeddings = read_embeddings(f_path)
    features = embeddings[set_indices:]
    return features


def read_embeddings(f_path):
    embeddings = []
    node_ids = []
    nnodes, dim = -1, -1
    with open(f_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                nnodes, dim = [int(s) for s in line.strip().split()[:2]]
            else:
                embedding = [float(s) for s in line.strip().split()[:1]]
                embeddings.append(np.array(embedding[1:]))
                node_ids.append(int(embedding[0]))
    embeddings = np.stack(embeddings)
    node_ids = np.array(node_ids)
    assert(embeddings.shape[0] == nnodes)
    assert (embeddings.shape[1] == dim)
    assert(len(np.unique(node_ids)) == nnodes)
    assert (node_ids.max() == nnodes-1)
    embeddings = embeddings[node_ids, :]
    return embeddings
