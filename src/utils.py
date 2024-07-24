import csv
import ast
import json
import os
import glob
import numpy as np
import re
from omegaconf import DictConfig, OmegaConf
from typing import Callable, List, Union, Tuple
import torch
from torch import Tensor, LongTensor
from src.model_nbf_general import NBFCluttr, compute_sim, entropy
from torch.optim import Optimizer
from torch.nn import Module
import random
import logging
import pickle
import math
from torch_scatter import scatter_sum
from dataclasses import dataclass
import networkx as nx
from torch_geometric.utils import degree, cumsum
log = logging.getLogger(__name__)

@dataclass
class Batcher:
	num_nodes: int
	target_edge_index: LongTensor
	edge_index: LongTensor
	edge_type: LongTensor
	target_edge_type: LongTensor
	graph_index: LongTensor = None
	graph_sizes: LongTensor = None


class PathError(Exception):
	"""Raised when the path is invalid"""
	pass

def get_acc(logits: Tensor, target_labels: Tensor):
    return torch.eq(logits.argmax(axis=1), target_labels).sum()/logits.shape[0]

def get_doubly_stochasic_tensor(
		*shape: Union[tuple[int], List[int]]
		) -> Tensor:
	if isinstance(shape[0], List) or isinstance(shape[0], tuple):
		shape = shape[0]
	for i in range(len(shape)):
		assert shape[0] == shape[i], 'need all tensor dims to be identical for this func'
	x = torch.rand(*shape)
	x = apply_prob_constraints_row_col(x)
	return x

def apply_prob_constraints_row_col(x: Tensor) -> Tensor:
	x = x / x.sum (dim = -1).unsqueeze (-1)
	if len(x.shape) > 1:
		x = x + (1 - x.sum (dim = -2).unsqueeze (-2)) / x.shape[-2]
	return x

def entropy_diff_agg(prob_T: Tensor, index: Tensor, num_nodes: int) -> Tensor:
	"""
	Entropic attention aggregation for probability vectors without using softmax.
	Essentially returns: 
		a_j = ∑_{i ∈ Neighbours(j)} (max-H -  H(p_i))/Z
	
	where Z is the normalizer over the neighbours. In effect, we are weighing the
	the contributions of each neighbour as a fraction of 1. The operation should be 
	analogous to `scatter_softmax`.

	Parameters
	----------
	prob_T : Tensor
		Tensor of arbitrary shape containing probability vectors as the final dimension.
	index : Tensor
		Aggregating indices to be summed over. Should be 1-D. 

	Returns
	-------
	Tensor
		reduced scalar coefficients for each p_i in `prob_T` normalized such that 
		summing over the neighbouring nodes in `index` is unity.
	"""
	prob_dim = prob_T.shape[-1]
	max_ent = - prob_dim * math.log(1./prob_dim) * 1./prob_dim 
	T_ent = entropy(prob_T, axis=-1)
	diff = max_ent-T_ent
	# aggregate over the appropriate neighbouring nodes / index
	diff_agg = scatter_sum(diff, index, dim=0, dim_size=num_nodes)
	diff /= diff_agg[index]
	assert diff.shape == prob_T.shape[:-1], 'need shape compatibility with `prob_T` to aggr. later.'
	return diff

def save_model(model: Module, epoch: int, opt: Optimizer, exp_name: str = None, 
			   model_path: str = None) -> None:
	state = {
	"model": model.state_dict(),
	"optimizer": opt.state_dict()
	}
	if not model_path:
		model_path = f"../models/{exp_name}_model_epoch_{epoch}.pth"
	torch.save(state, model_path)

def load_model_state(model_skeleton: Module, model_str: str, optimizer: Optimizer):
	model = model_skeleton
	if os.path.exists(model_str):
		state = torch.load(model_str)
	else:
		raise PathError(f"Model {model_str} does not exist.")

	log.info(f"Loading {model_str} from checkpoint.")
	try:
		model.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])
	except RuntimeError:
		# extra params that didn't exist in the old models are ignored
		pass


def save_json(data: dict, fname: str) -> None:
	with open(f"{fname}", 'w') as f:
		json.dump(data, f)

def remove_not_best_models(exp_name, best_epoch):
	models = glob.glob(f"../models/{exp_name}_model_epoch_*.pth")
	# cleanse the paths
	path_cleanser = lambda p: re.search('[0-9a-zA-Z_.]+.pth', p).group()
	models_cleansed = [path_cleanser(model) for model in models]  
	best_model = f"{exp_name}_model_epoch_{best_epoch}.pth"
	for model, c_model in zip(models, models_cleansed):
		if c_model != best_model:
			os.remove(model)

def get_most_recent_model_str(exp_name: str):
	"NOTE: internal function. best just use `load_model_state` directly"
	models = glob.glob(f"../models/{exp_name}_model_epoch_**")
	r = re.compile("(\d+)\.pth")
	models.sort(key=lambda x: int(r.search(x).group(1)))
	model_str=None
	if len(models) == 0:
		log.info("No checkpoint found. Training afresh.")
	else:
		if len(models) == 1: 
			# There should only be the best epoch model checkpoint
			model_str = models[0]
		else:
			# load from the most recent checkpoint
			model_str = models[-1]
	return model_str

def mkdirs(path: Union[str, List[str]]) -> None:
	if isinstance(path, str):
		os.makedirs(path, exist_ok=True)
	elif isinstance(path, List):
		for p in path:
			mkdirs(p)
	else:
		raise TypeError(f"Path {path} should be a string or a list of strings.")
	
def save_array(some_array, results_dir, exp_name, fname):
	save_dir = results_dir+f"/{exp_name}"
	mkdirs(save_dir)
	pickle.dump(some_array, open(save_dir+f"/{fname}.pkl", "wb"))

def save_results_models_config(config: DictConfig, exp_name: str, results_dir: str, 
								model_stuff: List, results_stuff: dict) -> None:
	save_dir = results_dir+f"/{exp_name}"
	mkdirs(save_dir)
	OmegaConf.save(config, save_dir+f"/config.yaml")
	save_model(*model_stuff, model_path=save_dir+f"/{exp_name}_model.pth")
	save_json(results_stuff, save_dir+f"/{exp_name}_results.json")
	
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
	
def read_datafile(filename, remove_not_chains=False):
	edge_ls = []
	edge_labels_ls = []
	query_edge_ls = []
	query_label_ls = []
	true_count = 0
	with open(filename, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			true_count += 1
			edges = row['story_edges']
			edges = ast.literal_eval(edges)
			edge_labels = ast.literal_eval(row['edge_types'])
			query_edge = ast.literal_eval(row['query_edge'])
			query_label = row['target']
			is_chain=True
			if remove_not_chains:
				for i in range(len(edges)-1):
					edge_i = edges[i]
					edge_j = edges[i+1]
					if edge_i[0] + 1 != edge_j[0] and edge_i[1] + 1 != edge_j[1]:
						is_chain=False
						break
			if not is_chain:
				continue
			edge_ls.append(edges)
			edge_labels_ls.append(edge_labels)
			query_edge_ls.append(query_edge)
			query_label_ls.append(query_label)

	data = {'edges':edge_ls,'edge_labels':edge_labels_ls,'query_edge':query_edge_ls,'query_label':query_label_ls}

	log.info(f"loaded {filename}: {len(data)} instances.")
	if remove_not_chains:
		log.info(f"removed {true_count - len(edge_ls)}/{true_count} not-chains.")
	return data

def find_unique_edge_labels(ls):
	unique = []
	for labels in ls:
		unique.extend(labels)
	unique = list(set(unique))
	return unique

def edge_labels_to_indices(ls,unique=None):
	if unique is None:
		unique = find_unique_edge_labels(ls)

	relabeled = [list(map(lambda y: unique.index(y),x)) for x in ls]
	return relabeled, unique
def query_labels_to_indices(ls,unique=None):
	if unique is None:
		unique = list(set(ls))
	
	relabeled = list(map(unique.index,ls))
	return relabeled, unique

def get_shortest_path_indices_from_edge_index(edge_index: Tensor, 
					 				   		  target_edge_index: Tensor
											  ) -> Union[List, List]:
	assert len(edge_index.shape) == 2, "edge_index should be should be two dimensional"
	assert len(target_edge_index.shape) == 2, "target_edge_index should be two dimensional"
	G = nx.Graph()
	if edge_index.shape[0] == 2:
		edge_index= edge_index.permute(1,0).detach().cpu().numpy()
	if target_edge_index.shape[0] == 2:
		target_edge_index= target_edge_index.permute(1,0).detach().cpu().numpy()
	G.add_edges_from(list(map(lambda x: tuple(x), edge_index)))
	all_pairs_shortest_paths = list(nx.all_pairs_bellman_ford_path(G))
	
	path_indices_union = []  
	agg_index_union = [] 
	for i, (source, target) in enumerate(target_edge_index):
		source, target = source.item(), target.item()
		source_to_sink_shortest_path = all_pairs_shortest_paths[source][1][target]
		path_indices_union.extend(source_to_sink_shortest_path)
		agg_index_union.extend([i]*len(source_to_sink_shortest_path))
	return path_indices_union, agg_index_union


def chain_edges(k: int, 
				b: int=1, 
				add_s_to_t_edge:bool = False,
				source_offset:int=0,
				node_offset:int=0,
				tail_node_tag: str = None,
				) -> List[Tuple[int, int]]:
	"""
	Generate a multi-chain graph edge list with multiple paths (chains) from a single 
	source to a single sink. Nodes are marked by `int`s from `0` to `k*b-1`.

	Parameters
	----------
	k : int
		number of edges in a path
	b : int, optional
		_description_, by default 1
	source_offset: int, optional
		start counting the source node from `source_offset`, by default 0
	node_offset: int, optional
		add an offset to the second node just after the source node, by default 0
	tail_node_tag: str, optional
		replace the tail node instances with a string instead of a value.
	add_s_to_t_edge : bool, optional
		add an edge connecting the source and sink nodes, by default False

	Returns
	-------
	List[Tuple[int, int]]
		an `edge_list`
	"""
    # make k-2 length chains and attach source and sink edges
	edge_list = []
	tail_node = (k-1)*b+2 - 1 + source_offset + node_offset
	source_node = source_offset
	current_node = source_offset + node_offset
	if tail_node_tag:
		tail_node_label = tail_node_tag
	else:
		tail_node_label = tail_node
	
	for _ in range(b):
		current_node += 1
		if current_node != source_offset:
			edge_list.append((source_node, current_node))
		intermediate_path_length = 0
		while intermediate_path_length < k-2:
			edge_list.append((current_node, current_node+1))
			intermediate_path_length  += 1
			current_node += 1
		if current_node != tail_node and current_node < tail_node:
			edge_list.append((current_node, tail_node_label))
	if add_s_to_t_edge:
		edge_list.append((0, tail_node))
	if add_s_to_t_edge:
		edge_list.append((0, tail_node))
	return edge_list, source_node, tail_node

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    print("Loaded {} records from {}".format(len(data), input_path))
    return data


def preprocess_graphlog_dataset(data_path: str) -> dict:

	data = load_jsonl(data_path)
	dataset = {'edges': [], 
			'edge_labels': [],
			'query_edge': [],
			'query_label': []
			}
	for graph in data:
		edge_index = []
		edge_labels = []
		edges = graph['edges']
		# deal with the train portion
		for edge in edges:
			node1, node2, label = edge
			edge_index.append((node1, node2))
			edge_labels.append(label)
		dataset['edges'].append(edge_index)
		dataset['edge_labels'].append(edge_labels)
		# deal with the query portion
		qnode1, qnode2, qlabel = graph['query']
		dataset['query_edge'].append((qnode1, qnode2))
		dataset['query_label'].append(qlabel)
	return dataset

def check_sub_graph(label, depth=3):
    if isinstance(label, list):
        return check_sub_graph(label[0], depth-1)
    if depth == 0:
        if isinstance(label, str):
            return True
    return False

def compute_path_length(path, depth=3):
    l = 0
    for edge_label in path:
        if check_sub_graph(edge_label, depth=depth):
            l += len(edge_label[0]) # add sub-path length
        else:
            l += 1
    return l

def make_graph_edge_list(graph_labels, depth=3):
    edge_list = []
    current_node = 0
    tail_node_tag = None
    for pn, path in enumerate(graph_labels):
        path_length = compute_path_length(path, depth)
        path_idx = 0
        for graph_label in path:
            if check_sub_graph(graph_label, depth):
                branches, sub_path_length = len(graph_label), len(graph_label[0])
                # use a global tail node tag to infer the tail node subsitution later in the game
                if path_idx + sub_path_length == path_length:
                    tail_node_tag = 'T'
                else:
                    tail_node_tag = None
                # main offset to the source node of the subgraph if it doesn't start from the global source node
                source_offset = current_node if  path_idx != 0 else 0
                # only if this is not the first subgraph emanating from the global source node
                node_offset = current_node if source_offset == 0 else 0
                
                sub_edge_list, _, _ = chain_edges(k=sub_path_length, b=branches, source_offset=source_offset, 
                                                  node_offset=node_offset, tail_node_tag=tail_node_tag)
                edge_list.extend(sub_edge_list)
                # print('sub_edge_list', sub_edge_list, 'source_offset', source_offset, 'node_offset', node_offset)
                # print('current_node', current_node)
                current_node = sub_edge_list[-1][1] if not isinstance(sub_edge_list[-1][1], str) else sub_edge_list[-1][0]
                path_idx += sub_path_length
            else:
                # tail node case
                if path_idx == path_length-1:
                    edge_list.append((current_node, 'T'))
                    current_node += 1
                    # tail node case a: pesky node offset for sub graph requires no update to the current node
                    if pn+1 < len(graph_labels):
                        if check_sub_graph(graph_labels[pn+1][0], depth):
                            current_node -= 1
                # source node case
                elif path_idx == 0:
                    if current_node == 0:
                        current_node += 1
                    if tail_node_tag:
                        current_node += 1
                    edge_list.append((0, current_node))
                else:
                    edge_list.append((current_node, current_node+1))
                    current_node += 1
                path_idx += 1
    # replace tagged tail node with the max node number
    max_edge_node = edge_list[-1][0] + 1
    for i in range(len(edge_list)):
        edge = edge_list[i]
        if edge[-1] == 'T':
            edge_list[i] = (edge[0], max_edge_node)
    return edge_list    

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def get_lr(optimizer):
	# https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
    for param_group in optimizer.param_groups:
        return param_group['lr']
	

def load_rcc8_file_as_dict(train_fname) -> dict:
	edge_ls = []
	edge_labels_ls = []
	query_edge_ls = []
	query_label_ls = []

	with open(train_fname, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			edges = ast.literal_eval(row['edges'])
			edge_labels = ast.literal_eval(row['edge_labels'])
			query_edge = ast.literal_eval(row['query_edge'])
			query_label = row['query_label']

			edge_ls.append(edges)
			edge_labels_ls.append(edge_labels)
			query_edge_ls.append(query_edge)
			query_label_ls.append(query_label)
	data = {'edges':edge_ls,'edge_labels':edge_labels_ls,'query_edge':query_edge_ls,'query_label':query_label_ls}
	print(f"loaded {train_fname}: {len(data)} instances.")
	return data

def get_sizes_to_unbatch_edge_index(
    edge_index: Tensor,
    batch: Tensor,
    batch_size: int = None,
) -> List[Tensor]:
	# taken from the `unbatch_edge_index` function in torch_geometric
	deg = degree(batch, batch_size, dtype=torch.long)
	ptr = cumsum(deg)

	edge_batch = batch[edge_index[0]]
	edge_index = edge_index - ptr[edge_batch]
	sizes = degree(edge_batch, batch_size, dtype=torch.long).cpu().tolist()
	return sizes