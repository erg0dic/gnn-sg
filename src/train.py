"""
Cluttr dataset preprocessing code is adapted from https://github.com/bergen/EdgeTransformer
"""
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from torch import nn
from torch_sparse import add_
from src.model_nbf_general import entropy, NBF, NBFCluttr
from src.model_nbf_fb import NBFdistR, NBFdistRModule, get_margin_loss_term, margin_loss
from typing import Union, List, Callable
import os
import numpy as np
import pickle
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import re
import wandb
from src.utils import (
    edge_labels_to_indices,
    load_jsonl,
    load_rcc8_file_as_dict, 
	query_labels_to_indices, 
	mkdirs, get_most_recent_model_str, 
	remove_not_best_models,
	save_results_models_config,
	get_temp_schedule,
	get_acc,
	read_datafile,
	save_model, 
	load_model_state,
	log, 
	save_array, 
	compute_sim,
	find_unique_edge_labels,
	preprocess_graphlog_dataset,
	get_lr,
	load_rcc8_file_as_dict,
	set_seed
)
from sklearn.metrics import confusion_matrix  
# # torch.set_deterministic_debug_mode(True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.set_float32_matmul_precision('medium')

def load_hydra_conf_as_standalone(config_name='config'):
	"NOTE: path needs to be relative to the configs directory."
	# https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
	config_path = "../configs"
	with initialize(version_base=None, config_path=config_path):
		cfg = compose(config_name=config_name)
	return cfg

def get_NBF_type(config_str: str) -> Union[Callable, NBF]:
	if config_str == 'NBF':
		return NBF
	elif config_str == 'NBFdistR':
		return NBFdistR
	else:
		raise NotImplementedError(f"Model {config_str} not implemented.")


class ClutrrDataset(Dataset):
	def __init__(self, dataset, reverse=False, fp_bp=False, 
			     unique_edge_labels=None,unique_query_labels=None):
		super().__init__()
		self.fp_bp = fp_bp
		self.edges = dataset['edges']

		#  unify unique edge and query labels
		if unique_edge_labels is None:
			unique_edge_labels = set(find_unique_edge_labels(dataset['edge_labels'])) 
			unique_query_labels = set(dataset['query_label'])
			unique_labels = list(unique_edge_labels.union(unique_query_labels))
		else:
			unique_labels = unique_edge_labels
			assert unique_edge_labels == unique_query_labels
		# assert len(unique_labels) == 20

		self.edge_labels, unique_edge_labels = edge_labels_to_indices(dataset['edge_labels'],unique_labels)
		self.query_edge = dataset['query_edge']
		self.query_label, unique_query_labels = query_labels_to_indices(dataset['query_label'],unique_labels)
		
		self.unique_edge_labels = unique_edge_labels
		self.unique_query_labels = unique_query_labels
		self.num_edge_labels = len(unique_edge_labels)
		self.num_query_labels = len(unique_query_labels)

		# consider edge reversal cases
		if reverse:
			self.edges, self.edge_labels = self.get_reversed_edges()
		# for simultaneous forward and backward pass
		if self.fp_bp:
			self.rev_edges, self.rev_edge_labels = self.get_reversed_edges()

	def  get_reversed_edges(self):
		rev_edges = list(map(lambda x: reverse_edges(x), self.edges))
		rev_edge_labels = list(map(lambda x: x[::-1], self.edge_labels))
		return rev_edges, rev_edge_labels
	
	def __len__(self):
		return len(self.edges)

	def __getitem__(self,index):	
		item = {
			'edge_index': torch.LongTensor(self.edges[index]).permute(1,0),
			'edge_type': torch.LongTensor(self.edge_labels[index]),
			'target_edge_index': torch.LongTensor(self.query_edge[index]).unsqueeze(1),
			'target_edge_type': torch.LongTensor([self.query_label[index]]),
		}
		if self.fp_bp:
			item['rev_edge_index'] = torch.LongTensor(self.rev_edges[index]).permute(1,0)
			item['rev_edge_type'] = torch.LongTensor(self.rev_edge_labels[index])
		return item
	
def remove_last_k_path(fname, k=1):
	splitf = fname.split('/')
	if k == 1:
		head, tail = splitf[:-k], splitf[-1]
	else:
		head, tail = splitf[:-k], splitf[-k:]
	return '/'.join(head), tail
	
def make_geo_transform(dataset, fp_bp=False):
	if fp_bp:
		return [HeteroData(
						fw = {'x':torch.arange(c['edge_index'].max().item()+1).unsqueeze(1) },
						bw = {'x':torch.arange(c['rev_edge_index'].max().item()+1).unsqueeze(1) },
						fw__rel__fw={
								'edge_index':c['edge_index'], 
								'edge_type':c['edge_type'], 
								'target_edge_index':c['target_edge_index'], 
								'target_edge_type':c['target_edge_type'], 
								}, 
						bw__rel__bw={
								'edge_index':c['rev_edge_index'], 
								'edge_type':c['rev_edge_type'], 
								}, 
						)  for c in dataset
				]
		
	else:
		return [Data(edge_index=c['edge_index'], 
			   		edge_type=c['edge_type'], 
					target_edge_index=c['target_edge_index'], 
					target_edge_type=c['target_edge_type'], 
					x=torch.arange(c['edge_index'].max().item()+1).unsqueeze(1)
					) for c in dataset
				]

def get_pickle_filename(fname, remove_not_chains=False, add_prefix=True, k=1):
	if ".." == fname[:2]:
		fname = fname[3:]
	path, file = remove_last_k_path(fname, k=k)
	assert isinstance(file, str), "File should be a string."
	pickle_path =path+"/pickles/"
	if add_prefix:
		pickle_path = '../'+pickle_path
	if not os.path.exists(pickle_path):
		os.mkdir(pickle_path)
	pfname = pickle_path + f"{file}_chains_{remove_not_chains}.pkl"
	return pfname

def reverse_edges(edge_list):
    sources = [x[0] for x in edge_list]
    sinks = [x[1] for x in edge_list]
    num_nodes = max(sources + sinks)
    reversed_nodes = list(range(num_nodes+1))[::-1]
    # map source to sink in edge_list
    new_edges = []
    for edge in edge_list:
        new_edge = reversed_nodes[edge[1]], reversed_nodes[edge[0]]
        new_edges.append(new_edge)
    return new_edges[::-1]

def get_data_loaders(fname: Union[List[str], str] = '../data/data_9b2173cf/1.2,1.3_train.csv', 
					batch_size=128, remove_not_chains=False, reverse=False, 
					fp_bp=False, dataset_type='clutrr'):
	# torch.manual_seed(42)

	# sanity test to make sure reverse isn't being used with fp_bp
	assert not(reverse and fp_bp), "reverse is incompatible with fp_bp. But got both True"
	if dataset_type != 'graphlog':
		fname_arg = fname 
	else: 
		fname_arg = fname[0]
	
	pfname = get_pickle_filename(fname_arg, remove_not_chains=remove_not_chains, k=1)
	if fp_bp:
		# pickle tag mod
		pfname = pfname.replace(".pkl", "_fp_bp.pkl")
	
	if not os.path.exists(pfname):
		if dataset_type == 'clutrr':
			data = read_datafile(fname, remove_not_chains=remove_not_chains)
			cdata = ClutrrDataset(data, reverse, fp_bp, None, None)
			pickle.dump(cdata, open(pfname, 'wb'))
			log.info(f"saving preprocessed data file at: {pfname}")
		elif dataset_type == 'graphlog':
			assert len(fname) == 2, 'need train and val files to be separate'
			fname_train, fname_val = fname[0], fname[1]
			data_train = preprocess_graphlog_dataset(fname_train)
			data_val = preprocess_graphlog_dataset(fname_val)
			cdata_train = ClutrrDataset(data_train, reverse, fp_bp, None, None)
			cdata_val = ClutrrDataset(data_val, reverse, fp_bp, cdata_train.unique_edge_labels, cdata_train.unique_query_labels)
			cdata = [cdata_train, cdata_val]
			pickle.dump(cdata, open(pfname, 'wb'))
			log.info(f"saving preprocessed data file at: {pfname}")
		elif dataset_type == 'rcc8':
			data = load_rcc8_file_as_dict(fname)
			cdata = ClutrrDataset(data, reverse, fp_bp, None, None)
			pickle.dump(cdata, open(pfname, 'wb'))
			log.info(f"saving preprocessed data file at: {pfname}")
		else:
			raise NotImplementedError
	else:
		log.info(f"preprocessed data file loaded from: {pfname}")
		cdata = pickle.load(open(pfname, 'rb'))
	return make_data_loaders(cdata, batch_size, train_ratio=0.8, fp_bp=fp_bp, dataset_type=dataset_type)

def get_dataset_test(fname: str = '../data/data_9b2173cf/1.3_test.csv',
                     unique_edge_labels=None, 
					 unique_query_labels=None, 
					 remove_not_chains=False, 
					 reverse=False, 
					 fp_bp=False, 
					 dataset_type='clutrr', 
					 batch_size=128):
	pfname = get_pickle_filename(fname, remove_not_chains=remove_not_chains)
	if fp_bp:
		# pickle tag mod
		pfname = pfname.replace(".pkl", "_fp_bp.pkl")
	if not os.path.exists(pfname):
		if dataset_type == 'clutrr':
			data = read_datafile(fname, remove_not_chains=remove_not_chains)
			cdata = ClutrrDataset(data, reverse, fp_bp, unique_edge_labels, unique_query_labels)
		elif dataset_type == 'graphlog':
			data = preprocess_graphlog_dataset(fname)
			cdata = ClutrrDataset(data, reverse, fp_bp, unique_edge_labels, unique_query_labels)
			assert cdata.unique_edge_labels == unique_edge_labels
		elif dataset_type == 'rcc8':
			data = load_rcc8_file_as_dict(fname)
			cdata = ClutrrDataset(data, reverse, fp_bp, unique_edge_labels, unique_query_labels)
		pickle.dump(cdata, open(pfname, 'wb'))
	else:
		cdata = pickle.load(open(pfname, 'rb'))
	# both of the following operations should preserve row order
	# dataset_geodic = make_geo_transform(cdata, fp_bp=fp_bp) # deterministic function: just containerize `cdata`
	test_dataset = make_geo_transform(cdata, fp_bp=fp_bp)
	if dataset_type == 'clutrr':
		test_dataset = Batch.from_data_list(test_dataset) # another wrapper
	return test_dataset

def make_data_loaders(cdata: Union[List[ClutrrDataset], ClutrrDataset], 
					  batch_size, train_ratio: float = 0.8, fp_bp=False, dataset_type='clutrr'):
	assert 0. < train_ratio <= 1., "acceptable domain is (0, 1]"
	if dataset_type in ['clutrr', 'rcc8']:
		train_dataset, val_dataset = random_split(cdata, [train_ratio, 1-train_ratio])
		unique_edge_labels = cdata.unique_edge_labels
		unique_query_labels = cdata.unique_query_labels
	elif dataset_type == 'graphlog':
		train_dataset, val_dataset = cdata[0], cdata[1]
		unique_edge_labels = train_dataset.unique_edge_labels
		unique_query_labels = train_dataset.unique_query_labels
		assert train_dataset.unique_query_labels == val_dataset.unique_query_labels
	else:
		raise NotImplementedError

	train_dataset, val_dataset = make_geo_transform(train_dataset,fp_bp=fp_bp), make_geo_transform(val_dataset, fp_bp=fp_bp)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size if dataset_type=='graphlog' else len(val_dataset), shuffle=True)

	return train_loader, val_loader, unique_edge_labels, unique_query_labels


def eval_model(model, test_dataset, fp_bp=False, **kwargs):
	mkdirs('../results')
	model.eval()
	logits, _ = model(test_dataset, **kwargs)
	if fp_bp:
		target_edge_type = test_dataset['fw', 'rel', 'fw'].target_edge_type
	else:
		target_edge_type = test_dataset.target_edge_type
	acc = get_acc(logits, target_edge_type)
	return acc.item()

def get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
						bw_only, final_linear, use_margin_loss, infer, 
						outs_as_left_arg, score_fn):
	
	num_data_points = 0.
	corrects = 0.
	
	for batch in dataset_test_loader:
		acc = eval_model(model, batch.to(device), fp_bp=fp_bp, 
								fw_only=fw_only, bw_only=bw_only, 
								final_linear=final_linear, 
								use_margin_loss=use_margin_loss,
								infer=infer, outs_as_left_arg=outs_as_left_arg,
								score_fn=score_fn)
		if fp_bp:
			target_edge_type = batch['fw', 'rel', 'fw'].target_edge_type
		else:
			target_edge_type = batch.target_edge_type
		num_data_points += target_edge_type.shape[0]
		corrects += acc*target_edge_type.shape[0]
	test_acc = corrects/num_data_points
	return test_acc

def get_test_metrics(data_train_path, 
					 unique_edge_labels,
					 unique_query_labels,
					 remove_not_chains,
					 bw_only,
					 model, final_linear,
					 fp_bp, infer, fw_only,
					 use_margin_loss, 
					 outs_as_left_arg,
					 score_fn, 
					 dataset_type,
					 batch_size=None):
	test_out = []
	# test
	if dataset_type == 'clutrr':
		dataset_dir = re.search("data_[0-9a-f]+", data_train_path).group()
		for i in range(2, 10+1):
			dataset_test = get_dataset_test(f"../data/{dataset_dir}/1.{i}_test.csv", 
									unique_edge_labels, unique_query_labels, 
									remove_not_chains=remove_not_chains,
									fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
			dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

			test_acc = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
						bw_only, final_linear, use_margin_loss, infer, 
						outs_as_left_arg, score_fn)
			test_out.append(test_acc)
		print(test_out)
		try:
			table = wandb.Table(columns = ['k', 'acc'], 
								data=[[k, acc] for k, acc in zip(range(2, 10+1), test_out)])
			wandb.log(
					{
						'test_accs': wandb.plot.line(table, "k", "acc", title=f"train: {data_train_path.split('/')[-1]}")
				}
			)
		except Exception as e:
			print('printing Exception')
			print('------------------')
			print(e)
	elif dataset_type == 'graphlog':
		datatest_path = '/'.join(i for i in data_train_path[0].split('/')[:-1]) + '/valid.jsonl'
		dataset_test = get_dataset_test(datatest_path, 
								unique_edge_labels, unique_query_labels, 
								remove_not_chains=remove_not_chains,
								fp_bp=fp_bp, dataset_type=dataset_type, batch_size=batch_size)
		dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
		test_acc = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
							bw_only, final_linear, use_margin_loss, infer, 
							outs_as_left_arg, score_fn)
		test_out.append(test_acc)
		print(test_out)

	elif dataset_type == 'rcc8':
		test_out = {}
		for path_len in [2,3,4,5,6,7,8,9,10]:
			for brl in [1,2,3]:
				fname = f'test_rcc8_k_{path_len}_b_{brl}.csv'
				dataset_test = get_dataset_test(f"../data/rcc8/{fname}", 
										unique_edge_labels, unique_query_labels, 
										remove_not_chains=remove_not_chains,
										fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
				dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

				test_acc = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
							bw_only, final_linear, use_margin_loss, infer, 
							outs_as_left_arg, score_fn)
				
				test_out[str((path_len, brl))] = test_acc
				print(f'{(path_len, brl)}: {test_acc}')
		print(test_out)
	else:
		raise NotImplementedError(f'test support for {dataset_type} has not been implemented.')
	return test_out

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(config: DictConfig) -> None:
	# resolve the value interpolations in the config file
	OmegaConf.resolve(config)

	# create the relevant directories
	results_dir = '../results'
	models_dir = '../models'
	mkdirs([results_dir, models_dir])
	
	if config.turn_on_wandb:
	
		try:
			run = wandb.init(project=config.wandb.project, name=config.experiments.exp_name, id=config.experiments.exp_name)
		except Exception as e:
			print(e)
	
	# load all the experimental config parameters from `config`
	NBF_type, epochs, exp_name = get_NBF_type(config.experiments.NBF_type), config.experiments.epochs, config.experiments.exp_name
	remove_not_chains = config.experiments.remove_not_chains
	data_train_path = config.experiments.data_train_path
	hidden_dim = config.experiments.get('hidden_dim', config.hidden_dim)
	num_layers = config.num_layers
	eval_mode = config.experiments.get('eval_mode', False)

	fp_bp = config.experiments.get('fp_bp', False)
	facets = config.experiments.get('facets', 1)
	fw_only = config.experiments.get('fw_only', False)
	bw_only = config.experiments.get('bw_only', False)

	use_margin_loss = config.experiments.get('use_margin_loss', False)
	num_negative_samples = config.experiments.get('num_negative_samples', 10)
	margin = config.experiments.get('margin', 0.1)
	score_fn = config.experiments.get('score_fn', 'xent')
	final_linear = config.experiments.get('final_linear', False)
	outs_as_left_arg = config.experiments.get('outs_as_left_arg', True)
	infer = config.experiments.get('infer', True)
	dataset_type = config.experiments.get('dataset_type', 'clutrr')
	batch_size = config.experiments.get('batch_size', 128)
	rel_offset = config.experiments.get('rel_offset', 0)
	aggr_type = config.experiments.get('aggr_type', 'mul') 
	seed = config.experiments.get('seed', 42)

	# ablations
	ablate_compose = config.experiments.get('ablate_compose', False)
	ablate_probas = config.experiments.get('ablate_probas', False)
	set_seed(seed)
	 


	# fixed stuff follows here:
	lr = config.experiments.get('lr', 0.001)
	if config.set_hidden_eq_num_relations:
		hidden_dim = len(unique_query_labels)
		log.warn(f"Hidden dim {hidden_dim} is the same as number of relations")

	train_loader, val_loader, unique_edge_labels, \
	unique_query_labels = get_data_loaders(remove_not_chains=remove_not_chains, 
										         fname=data_train_path, fp_bp=fp_bp, 
												 dataset_type=dataset_type, batch_size=batch_size)
	loss = nn.CrossEntropyLoss()
	model = NBF_type(hidden_dim=hidden_dim, num_relations=len(unique_query_labels)+rel_offset, 
			 shared=config.shared, use_mlp_classifier=config.use_mlp_classifier, 
			 dist=config.dist, num_layers=num_layers,  
			 residual=False, 
			 eval_mode=eval_mode,
			 facets=facets,
			 aggr_type=aggr_type,
			 ablate_compose=ablate_compose,
			 ablate_probas=ablate_probas,
			 )


	model.to(device)
	opt = AdamW(model.parameters(), lr=lr)

	epoch=0
	if config.experiments.load_from_checkpoint:
		model_str = get_most_recent_model_str(exp_name)
		if model_str:
			load_model_state(model, model_str, opt)
			match = re.search("model_epoch_[0-9]+", model_str)
			if match:
				epoch = int(match.group().split("_")[-1])
				assert epoch < epochs, f"Epochs {epochs} should be greater than the loaded epoch {epoch}."
	else:
		log.info("Training afresh. WARNING: previous checkpoints will be overwritten.")
	best_acc = 0
	best_epoch = epoch
	# model.train()
	accs = []
	all_epoch_ys = []
	
	
	while epoch < epochs:
		epoch_train_losses = []
		# train
		for batch in train_loader:
			batch = batch.to(device)
			if fp_bp:
				target_edge_type = batch['fw', 'rel', 'fw'].target_edge_type
			else:
				target_edge_type = batch.target_edge_type
			opt.zero_grad()
			if use_margin_loss:
				outs_bfh, rs_rfh = model(batch, fw_only=fw_only, bw_only=bw_only, use_margin_loss=True, final_linear=final_linear)
				loss_train = get_margin_loss_term(outs_bfh, rs_rfh, target_edge_type, 
												  num_negative_samples=num_negative_samples, 
												  margin=margin, score_fn=score_fn, 
												  outs_as_left_arg=outs_as_left_arg)

			else:
				logits, proto_proba = model(batch, fw_only=fw_only, bw_only=bw_only, final_linear=final_linear)
				loss_train = loss(logits, target_edge_type)

			epoch_train_losses.append(loss_train.item())
			loss_train.backward()
			opt.step()
		# validate
		#  TODO: add early stopping?
		with torch.no_grad():
			num_data_points = 0.
			corrects = 0.
			loss_margin_tot = 0.
			for batch in val_loader:
				batch = batch.to(device)
				if use_margin_loss:
					assert infer, "Margin loss requires that inference be made using the score function."
					outs, _ = model(batch, fw_only=fw_only, bw_only=bw_only, 
					 				use_margin_loss=True, final_linear=final_linear,
									infer=True, outs_as_left_arg=outs_as_left_arg, score_fn=score_fn)
					
					outs_bfh, rs_rfh = model(batch, fw_only=fw_only, bw_only=bw_only, 
							                 use_margin_loss=True, final_linear=final_linear)
					if dataset_type != 'graphlog':
						print('outs', torch.softmax(outs[0], dim=-1)) 
					logits = outs
				else:
					logits, probas = model(batch, fw_only=fw_only, bw_only=bw_only, final_linear=final_linear)
				if fp_bp:
					target_edge_type = batch['fw', 'rel', 'fw'].target_edge_type
				else:
					target_edge_type = batch.target_edge_type

				loss_val = loss(logits, target_edge_type)
				loss_margin_val = get_margin_loss_term(outs_bfh, rs_rfh, target_edge_type, 
												  num_negative_samples=num_negative_samples, 
												  margin=margin, score_fn=score_fn, 
												  outs_as_left_arg=outs_as_left_arg).item() 

				val_acc = get_acc(logits, target_edge_type)
				num_data_points += target_edge_type.shape[0]
				corrects += val_acc*target_edge_type.shape[0]
				loss_margin_tot += loss_margin_val
			val_acc = corrects/num_data_points
			# save some stats
			accs.append(val_acc.item())	
			# TAG: scheduler
			# if dataset_type == 'graphlog':
			# 	scheduler.step()
		# save the best model for this epoch
			save_model(model, epoch, opt, exp_name)
		log.info(f"Epoch train {epoch} loss: {np.mean(epoch_train_losses)}")
		# log.info(f'aggr hparam: {torch.sigmoid(model.BF_layers[0].lambdaa).item()}')
		log.info(f"Epoch val {epoch} loss xent: {loss_val.item()}, acc: {val_acc.item()}, loss mar: {loss_margin_tot}")	
		if dataset_type != 'graphlog':
			log.info(f"Epoch {epoch} val confusion mat: \n{confusion_matrix(logits.detach().cpu().argmax(dim=-1), target_edge_type.detach().cpu())}")
		else:
			log.info(f'lr is: {get_lr(opt)}')
		all_epoch_ys.append(model.BF_layers[0].multi_embedding.detach().cpu().numpy())		
		

		if val_acc.item() > best_acc:
			best_acc = val_acc.item()
			best_epoch = epoch
		epoch += 1

	# TAG: wandb 
	if config.do_hyper_sweep:
		# only hypersweep on the training set as the test set is specifically for SG 
		wandb.log({"val_acc": val_acc.item()})
	# clean up
	# best_epoch=epochs-1
	remove_not_best_models(exp_name, best_epoch)
	# load the best model and change to eval mode
	load_model_state(model, f"../models/{exp_name}_model_epoch_{best_epoch}.pth", opt)
	if config.shared:
		model.eval_mode = True
		model.BF_layers[0].eval_mode = True
	# call dataset-agnostic test function 
	test_accs = get_test_metrics(data_train_path, unique_edge_labels, unique_query_labels,
					             remove_not_chains, bw_only, model, final_linear,
					 			 fp_bp, infer, fw_only, use_margin_loss, outs_as_left_arg,
								 score_fn, dataset_type, batch_size)
	# save results
	results_dict = dict()
	results_dict['test_accs'] = test_accs
	results_dict['accs'] = accs
	save_results_models_config(config, exp_name, results_dir, 
							   [model, best_epoch, opt], results_dict)

	save_array(all_epoch_ys, results_dir, exp_name, "all_epoch_ys")
	

if __name__ == '__main__':
	
	run()
