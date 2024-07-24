from functools import partial
from typing import Callable, Dict, List, Tuple, Set, Type
from src import train
from src.composition_tables.baseclass import SemiGroup
import numpy as np
from copy import deepcopy
import random
from src.utils import chain_edges, make_graph_edge_list, flatten
import pickle
import os

def make_instance(k, b, add_s_to_t_edge=False, 
                  semigroup=SemiGroup, 
                  translate: Callable = None):

   edge_list, _, tail_node = chain_edges(k=k, b=b, add_s_to_t_edge=add_s_to_t_edge)
   size = tail_node+1 # graph size
   csp = {}

   for v in range(size):
      j = {}   
      j[v] = 'EQ#unk' # placeholder eq
      csp[v] = j

   for i in edge_list:
      choice = random.choice(semigroup.elements)
      csp[i[0]][i[1]] = choice

   edge_list_labelled = []
   for node_i in csp:
      for node_j in csp[node_i]:
         if node_i != node_j:
            edge_list_labelled.append([translate(r) for r in [csp[node_i][node_j]]])

   return csp, edge_list_labelled


def compose(r1: List[Set[int]], 
            r2: List[Set[int]],
            composition_table: Dict[Tuple[int, int], List[int]]
            ) -> Set[int]:
    result = []
    for label1 in r1:
        for label2 in r2:
            result.extend(composition_table[(label1, label2)])
    return set(result)

def compute_algebraic_closure_for_paths(edge_list: List[List[int]], 
                                        composition_table: Dict[Tuple[int, int], List[int]] = None
                                        ) -> Set[int]:
    i = 0
    stack = [edge_list[0]]
    while i < len(edge_list) - 1:
        comps = compose(set(stack.pop(-1)), set(edge_list[i+1]), composition_table=composition_table)
        # note that a set is a subset of itself is the target base case here
        # i.e. the currents are not changing
        stack.append(comps)
        i += 1
    return stack[0]

def make_a_lot_of_paths(num_paths: int, k: int, b: int, 
                        str2int: Dict[str, int] = None,
                        make_instance: Callable = None,
                        composition_table: Dict[Tuple[int, int], List[int]] = None
                        ) -> Tuple[List[List[str]], List[Set[int]]]:
    "Generates paths and algebraic closures. Only some paths will yield singleton closures."
    # assert k >= 2, 'number of nodes needs to be at least 2 for an edge to exist.'
    paths = []
    path_closures = []
    for _ in range(num_paths):
        _, edge_list = make_instance(k, b)
        paths.append(edge_list)
        numerify_edge_list = [[str2int[edge] for edge in edges] for edges in edge_list]
        closure_i = compute_algebraic_closure_for_paths(numerify_edge_list, composition_table=composition_table)
        path_closures.append(closure_i)
    return paths, path_closures

def filter_paths_wrt_el(el: int, 
                        paths: List[Set[int]]
                        ) -> Tuple[List[Set[int]], np.ndarray]:
    out_paths = []
    filter_indices = []
    for i, path in enumerate(paths):
        if el in path:
            out_paths.append(path)
            filter_indices.append(i)
    return out_paths, np.array(filter_indices)


def get_path_filter_wrt_el(semigroup: SemiGroup) -> Dict[int, Callable]:
    path_closure_filter_wrt_el_dict = {}
    for el in semigroup.elements:
        path_closure_filter_wrt_el_dict[el] = partial(filter_paths_wrt_el, el)
    return path_closure_filter_wrt_el_dict

def make_base_case_graph_for_group(semigroup: SemiGroup,
                                    num_samples: int,
                                    num_branches: int,
                                    path_closures: List[List[int]]
                                    ) -> Dict[int, List[List[int]]]:
    """
    tl;dr: generates a skeleton graph with 2 edges and `num_branches` branches
    that will be expanded later. The graph has a singleton edge label between the
    source and the sink node.

    A picture that might help:
         ___o___
        /       \
        o - o - o
        \___o___/   
    where `num_branches` = 3

    Returns a filtering of paths based on `path_closures` such that the filtered
    paths all produce singletons. 
    
    The paths are organized by the target singleton they yield as the key and the index
    of the path in the original `path_closures` as the value.
    
    An example output looks like: {'el_1': [path_5, path_257], 
                                    ..., 
                                   'el_n': [path_14, path_420]}

    Parameters
    ----------
    semigroup : SemiGroup
        a `dataclass` struct containing the semi-group elements
    num_samples : int
        number of singleton paths per semi-group element to generate
    num_branches : int
        number of multiple branches per path to generate that under intersection 
        yield the target singleton element
    path_closures : List[List[int]]
        the composition algebra generated after composing all the edges in the path 
        and intersecting all branches per path.
    Returns
    -------
    Dict[int, List[List[int]]]
        A dictionary of lists of `num_samples` paths that yield the target singleton element as key.
    """
    # get the filtered paths for the el
    # work with edge labels instead of edges. So essentially just stacked chains 
    #     Graph = [chain_1=[label_1, label_2] 
    #               ..., 
    #              chain_{num_branches}=[label_1, label_2]
    #             ]
    filters = get_path_filter_wrt_el(semigroup)
    graphs = {el: np.zeros((num_samples, num_branches), dtype=np.int32) for el in semigroup.elements}
    for el in semigroup.elements:
        filtered_path_closures, filtered_path_closures_inds = filters[el](path_closures)
        for n in range(num_samples):
            while True:
                # random sample from the filtered path closures
                rel_path_closures_inds = np.random.randint(0, len(filtered_path_closures), size=num_branches)
                path_closures_inds = filtered_path_closures_inds[rel_path_closures_inds]
                # compute the algebraic closure
                closure_list = []
                for path_c_i in rel_path_closures_inds:
                    closure_i = filtered_path_closures[path_c_i]
                    closure_list.append(closure_i)
                while len(closure_list) > 1:
                    set1 = closure_list.pop()
                    set2 = closure_list.pop()
                    closure_list.append(set1.intersection(set2))
                if len(closure_list[0]) == 1 and list(closure_list[0])[0] == el:
                    graphs[el][n] = path_closures_inds
                    break
    return graphs

def intersect_sets(sets: List[Set[int]]) -> Set[int]:
    out = sets[0]
    for i in range(1,len(sets)):
        out = out.intersection(sets[i])
    return out


## Rationale:
# 1. Generate the simplest chain paths of length 3 for all singleton edge labels. 
# 2. Make the desired number of branches $k$ version of your graph. Keep path length fixed to 3. 
# 3. Pick edges randomly on each chain in all the branches and replace them with a random chain
#    path of length 3 generated at step 1. 
#     - recursive expansion 
#     - note that all edge labels are singletons 
#     - pick an edge singleton to expand on random 
#     - find the random 3-chain graph for that edge singleton 
#     - replace 
# 4. Keep repeating until 3. for each chain in the branch until the desired path length 
#    for that branch is achieved. Then repeat this for each branch.

# NOTE: **_more_diverse_** functions that follow after generalise the edge replacer graph
#       with more than 2 edge length paths. They are themselves generated by random sampling 
#       and not the aforementioned process.

def make_more_diverse_graphs_for_group(semigroup: SemiGroup,
                                        num_samples: int,
                                        num_branches: int,
                                        path_length: int,
                                        cache_3_chain_size: int=100000,
                                        make_instance: Callable = make_instance, # this is fixed 
                                        ) -> Dict[int, List[List[int]]]:
    "`path_length` is the number of nodes in a path"
    # paths will be modified in place and path closures should be invariant to the applied changes
    make_instance = partial(make_instance, semigroup=semigroup, translate=semigroup.translate, add_s_to_t_edge=False)
    # these data structs will be updated
    cache_path = 'composition_tables/caches/'
    fname = f'{cache_path}/diverse_group_graphs_{semigroup.name}_cache_{cache_3_chain_size}_{num_samples}_.pkl'
    os.makedirs(cache_path, exist_ok=True)

    if not os.path.exists(fname):
        paths, path_closures =  make_a_lot_of_paths(num_paths=cache_3_chain_size, k=2, b=1, make_instance=make_instance, 
                                                str2int=semigroup.str2int, composition_table=semigroup.composition_table)
        
        additional_paths = []
        additional_path_closures = []
        for edges in range(3, 4+1):
            add_path, add_path_closure = make_a_lot_of_paths(num_paths=cache_3_chain_size, k=edges, b=1, 
                        make_instance=make_instance, str2int=semigroup.str2int, composition_table=semigroup.composition_table)
            additional_paths.append(add_path)
            additional_path_closures.append(add_path_closure)
        
        slot_machine = []
        for path_cl in [path_closures, *additional_path_closures]:
            # find chains of `path_length`s above that end in singletons
            # for some reason edge_length 5 is impossibly for large sample sizes... TODO Why?
            slot_machine.append(make_base_case_graph_for_group(semigroup, num_samples, 1, path_cl))
            print('done with path_cl', len(slot_machine))
        
        paths_2_edges = deepcopy(paths)
        slot_machine_paths = [paths_2_edges, *additional_paths]
        pickle.dump((paths, path_closures, slot_machine, slot_machine_paths), open(fname, 'wb'))
    else:
        paths, path_closures, slot_machine, slot_machine_paths = pickle.load(open(fname, 'rb'))
    
    # filtered indices of `paths` that yield singletons. 
    # NOTE: only these indices are expanded in `paths` 
    graphs = make_base_case_graph_for_group(semigroup, num_samples, num_branches, path_closures)

    for el in semigroup.elements:
        graphs_per_el = graphs[el]
        for n in range(num_samples):
            for b in range(num_branches):
                # path to expand as an edge list
                path_nb = paths[graphs_per_el[n][b]]
                while len(path_nb) < path_length:
                    # pick an edge in the `path_nb` to expand
                    edge_to_expand_index = random.randint(0, len(path_nb)-1)
                    # convert group elements from str to int
                    edge_to_expand_el = semigroup.str2int[path_nb[edge_to_expand_index][0]]
                    # pick a random graph index
                    rgi = random.randint(0, num_samples-1) 

                    slot_machine_idx_max = min(len(slot_machine)-1, path_length-len(path_nb)-1)
                    slot = random.randint(0, slot_machine_idx_max)
                    singleton_path_idx = slot_machine[slot][edge_to_expand_el][rgi][0]
                    
                    path_to_expand = slot_machine_paths[slot][singleton_path_idx]
                
                    path_nb.pop(edge_to_expand_index)
                    for i in range(len(path_to_expand)):
                        path_nb.insert(edge_to_expand_index+i, path_to_expand[i])
    
                    assert len(path_to_expand) == slot+2, f'need a {slot+3}-chain represented by {slot+2} edge labels as an expanding path but got {len(path_to_expand)} edges instead. {breakpoint()}'
    return graphs, paths, path_closures



def make_diverse_graphs_with_recursive_branches_v2(semigroup,
                                                num_samples: int,
                                                num_branches: int,
                                                final_path_length: int,
                                                cache_3_chain_size: int=100000,
                                                make_instance = make_instance, 
                                                base_path_length=2
                                                ):
    "V2: adds sub-loops of arbitrary path length as opposed to just 2 length paths in v1"
    assert base_path_length >= final_path_length-base_path_length, 'this is the minimum number of graph expansion chances required at recursion depth of 1'
    # make a base graph that will be expanded
    graphs, paths, path_cls = make_more_diverse_graphs_for_group(semigroup, num_samples, num_branches, 
                                                    path_length=base_path_length, 
                                                    cache_3_chain_size=cache_3_chain_size, 
                                                    make_instance=make_instance)
    
    sampling_cache = {}
    path_lens = range(2, final_path_length - 2 + 1)
    MIN_BL, MAX_BL = 2, 5
    branch_lens = range(MIN_BL, MAX_BL+1)
    cache_path = 'composition_tables/caches/'
    os.makedirs(cache_path, exist_ok=True)
    for path_len in path_lens:
        for branch_len in branch_lens:
            print(f'path_len: {path_len}, branch_len: {branch_len}')
            fname = f'{cache_path}/diverse_recursive_graphs_{semigroup.name}_cache_{cache_3_chain_size}_{path_len}_{num_samples}_{branch_len}_.pkl'
            if not os.path.exists(fname):
                subgraphs, subpaths, _ = make_more_diverse_graphs_for_group(semigroup, num_samples, 
                                                                            branch_len, path_len,   
                                                                            cache_3_chain_size, make_instance)

                pickle.dump((subgraphs, subpaths), open(fname, 'wb'))
            else:
                subgraphs, subpaths = pickle.load(open(fname, 'rb')) 
            
            sampling_cache[(path_len, branch_len)] = (subgraphs, subpaths)
    
    # extra complexity due to repeated path hashes in `graphs` 
    seens = {}
    
    for el in semigroup.elements:
        for n in range(num_samples):
            for b in range(num_branches):
                path_hash_index = graphs[el][n][b]

                if path_hash_index in seens:
                    continue
                else:
                    seens[path_hash_index] = None

                path_nb = paths[path_hash_index]
                # don't expand an edge that is already expanded. 
                # Limit recursion to depth 1
                valid_edge_indices = [i for i in range(len(path_nb))]
                path_nb_length = len(path_nb)
                expansion_idxes = []
                # sanity test
                if path_nb_length >= final_path_length:
                    raise AssertionError(f'path length is already {path_nb_length} but target is {final_path_length}.')
                while path_nb_length < final_path_length:
                    edge_to_expand_index = random.choice(valid_edge_indices)
                    expansion_idxes.append(edge_to_expand_index)
                    valid_edge_indices.remove(edge_to_expand_index) 
                    edge_to_expand_el = semigroup.str2int[path_nb[edge_to_expand_index][0]]
                    # pick a random subgraph to add subject to path length constraints
                    diff = final_path_length - path_nb_length
                    sub_graph_pl_ix = random.randint(2, min(final_path_length - 2, diff + 1))
                    bl_ix = random.randint(MIN_BL, MAX_BL)
                    sub_graphs, sub_paths = sampling_cache[(sub_graph_pl_ix, bl_ix)] 

                    # pick a random sub graph index
                    rgi = random.randint(0, num_samples-1)
                    sub_graph_as_idx = sub_graphs[edge_to_expand_el][rgi]
                    sub_graph = [sub_paths[sub_graph_as_idx[sub_branch]] for sub_branch in range(bl_ix)]
                    path_nb.pop(edge_to_expand_index)
                    path_nb.insert(edge_to_expand_index, sub_graph)
                    path_nb_length += len(sub_graph[0])-1 # we only get <`diff`-1 additional edges
                # test final path length
                loop_invariant_path_length = len(path_nb)
                for idx in expansion_idxes:
                    loop_invariant_path_length += len(path_nb[idx][0])-1
                assert loop_invariant_path_length == final_path_length, f'target path length was {final_path_length} but got {len(path_nb)} instead. {breakpoint()}'

    return graphs, paths, path_cls

def make_dataset_for_comp_data(graphs, paths, semigroup, dataset_dict = None):
    keys = list(graphs.keys())
    num_samples = graphs[keys[0]].shape[0]
    num_branches = graphs[keys[0]].shape[1]
    int2str = {value:key for key,value in semigroup.str2int.items()}
    if not dataset_dict:
        dataset_dict = {
                'edges': [],
                'edge_labels': [],
                'query_edge': [],
                'query_label': [],
                
                }
    for el in semigroup.elements:
        for n in range(num_samples):
            edge_label_list = []
            for b in range(num_branches):
                edge_label_list.append(paths[graphs[el][n][b]])
            topology = make_graph_edge_list(edge_label_list, depth=3)
            query_edge = (0, topology[-1][-1])
            target_edge_type = int2str[el]
            flattened_edge_list = flatten(edge_label_list)
            dataset_dict['edges'].append(topology)
            dataset_dict['query_edge'].append(query_edge)
            dataset_dict['query_label'].append(target_edge_type)
            dataset_dict['edge_labels'].append(flattened_edge_list)
    return dataset_dict

if __name__ == '__main__':
    from src.composition_tables.rcc8 import RCC8
    from src.composition_tables.interval import Interval
    # graphs, paths, path_closures = make_diverse_graphs_with_recursive_branches(RCC8, 10, 3, 5, 
    #                                                          cache_3_chain_size=100000,
    #                                                          additional_edges=2,
    #                                                          )


    # SEMIGROUP = RCC8
    # NUM_SAMPLES = 20000
    # PATH_LENGTH = 10
    # cache_size = 1000000
    # graphs, paths, path_cls = make_diverse_graphs_with_recursive_branches_v2(SEMIGROUP, 
    #                                             NUM_SAMPLES, 5, PATH_LENGTH, cache_size, base_path_length=5)


    # Test set
    #         b=1,2,3 .. 4,5,6,7,8,9,10,11,12
    #         k=2,3,4 .. 5,6,7,8,9,10,11,12
    #         
    #         each will be treated separetely and will have 1000 samples 
    #         

    import pandas as pd

    def make_training_set(semigroup, num_samples, cache_size=1000000):
        # Train dataset
        #           b=1,2,3
        #           k=2,3,4
        # total size: 42000, and each example will be around 7000
        dataset_dict=None
        for PATH_LENGTH in [2,3,4]:
            for BRL in [1,2,3]:
                graphs, paths, path_cls = make_more_diverse_graphs_for_group(
                                                        semigroup, 
                                                        num_samples, 
                                                        BRL, 
                                                        PATH_LENGTH, 
                                                        cache_size)
                print(f'path-length, brl: {PATH_LENGTH}, {BRL}')
                dataset_dict = make_dataset_for_comp_data(graphs, paths, semigroup, dataset_dict=dataset_dict)
        return dataset_dict

    def make_and_deploy_test_set(semigroup, num_samples, cache_size=1000000):
        #  each k, brl case will need to be its own csv file for us to eval in a fine-grained manner
        data_dir = '../data/rcc8/'
        os.makedirs(data_dir, exist_ok=True)
        # repeat the train set case
        for PATH_LENGTH in [2,3,4]:
            for BRL in [1,2,3]:
                fname = data_dir+f'test_{semigroup.name}_k_{PATH_LENGTH}_b_{BRL}.csv'
                if not os.path.exists(fname):
                    graphs, paths, path_cls = make_more_diverse_graphs_for_group(
                                                            semigroup, 
                                                            num_samples, 
                                                            BRL, 
                                                            PATH_LENGTH, 
                                                            cache_size)
                    dataset_dict = make_dataset_for_comp_data(graphs, paths, semigroup, dataset_dict=None)
                    df = pd.DataFrame(dataset_dict)
                    df.to_csv(fname)
                    print(f'path-length, brl: {PATH_LENGTH}, {BRL}')

        for PATH_LENGTH in [5, 6, 7, 8, 9, 10, 11, 12]:
            for BRL in [1, 2, 3, 4, 5, 6, 7, 8]:
                fname = data_dir+f'test_{semigroup.name}_k_{PATH_LENGTH}_b_{BRL}.csv'
                if not os.path.exists(fname):
                    graphs, paths, path_cls = make_diverse_graphs_with_recursive_branches_v2(
                                                    semigroup,
                                                    num_samples,
                                                    BRL,
                                                    PATH_LENGTH,
                                                    cache_size, 
                                                    base_path_length=PATH_LENGTH//2+1                                
                                                    )
                    dataset_dict = make_dataset_for_comp_data(graphs, paths, semigroup, dataset_dict=None)
                    df = pd.DataFrame(dataset_dict)
                    df.to_csv(fname)
                    print(f'path-length, brl: {PATH_LENGTH}, {BRL}')



    def deploy_train_set():
        data_dir = '../data/rcc8/'
        os.makedirs(data_dir, exist_ok=True)
        semigroup = RCC8
        num_samples = 800 # per group element, k, brl so, dataset size is actually 8*9*800=57600
        train_dataset = make_training_set(semigroup, num_samples)
        
        df = pd.DataFrame(train_dataset)
        df.to_csv(data_dir+f'train_{semigroup.name}.csv')

    make_and_deploy_test_set(RCC8, 800)