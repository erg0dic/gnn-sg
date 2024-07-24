from src.composition_tables.rcc8 import RCC8
from src.composition_tables.interval import Interval
from src.comp_dataset_generator import (
    make_diverse_graphs_with_recursive_branches_v2,
    make_graphs_for_group, 
    compute_algebraic_closure_for_paths, 
    intersect_sets,
    make_more_diverse_graphs_for_group,
    intersect_sets, 
)
from src.utils import check_sub_graph
import pytest

@pytest.fixture
def get_rcc8_graphs():
    num_samples=100
    num_branches=5
    path_length=20
    graphs, paths, path_closures = make_graphs_for_group(RCC8, num_samples, num_branches, path_length, 
                                                         cache_3_chain_size=100000)
    return num_samples, num_branches, graphs, paths, path_closures

@pytest.fixture
def get_interval_graphs():
    num_samples=1000
    num_branches=5
    path_length=10
    graphs, paths, path_closures = make_graphs_for_group(Interval, num_samples, num_branches, path_length, 
                                                         cache_3_chain_size=100000)
    return num_samples, num_branches, graphs, paths, path_closures


def test_rcc8_datagen_correctness(get_rcc8_graphs):
    num_samples, num_branches, graphs, paths, path_closures = get_rcc8_graphs
    # test if path closure is still invariant
    _do_path_closure_loop_invariant_test(RCC8, num_samples, num_branches, graphs, paths, path_closures)

def test_interval_datagen_correctness(get_interval_graphs):
    num_samples, num_branches, graphs, paths, path_closures = get_interval_graphs
    # test if path closure is still invariant
    _do_path_closure_loop_invariant_test(Interval, num_samples, num_branches, graphs, paths, path_closures)

def test_rcc8_diverse_graphs():
    num_samples=10
    num_branches=5
    for path_length in [3, 4, 5, 6, 10]:
        graphs, paths, path_closures = make_more_diverse_graphs_for_group(RCC8, num_samples, num_branches, path_length, 
                                                            cache_3_chain_size=100000)
        _do_path_closure_loop_invariant_test(RCC8, num_samples, num_branches, graphs, paths, path_closures) 



def _do_path_closure_loop_invariant_test(semigroup, num_samples, num_branches, graphs, paths, path_closures):
    for el in semigroup.elements:
        graphs_per_el = graphs[el]
        for n in range(num_samples):
            new_path_closures = []
            for b in range(num_branches):
                path_as_int_labels = list(map(lambda x: [semigroup.str2int[x[0]]], paths[graphs_per_el[n][b]])) 
                new_path_closure = compute_algebraic_closure_for_paths(path_as_int_labels, 
                                                                       composition_table=semigroup.composition_table)
                old_path_closure = path_closures[graphs_per_el[n][b]]
                assert new_path_closure == old_path_closure, 'path closure should be invariant'
                new_path_closures.append(new_path_closure)
            closure_intersection = intersect_sets(new_path_closures)
            assert closure_intersection == {el}, 'closure intersection should be the target singleton'

# super crude but quick test to make sure that the dataset is sound by 
# just collapsing the sub loops, and later the main loop, 
# into a path consistency problem
def test_more_diverse_graphs_v2_correctness():
    SEMIGROUP = RCC8
    NUM_SAMPLES = 100
    PATH_LENGTH = 5
    cache_size = 10000
    graphs, paths, _ = make_diverse_graphs_with_recursive_branches_v2(SEMIGROUP, 
                                                    NUM_SAMPLES, 5, PATH_LENGTH, cache_size, 
                                                    base_path_length=3)
    keys = list(graphs.keys())
    num_samples = graphs[keys[0]].shape[0]
    num_branches = graphs[keys[0]].shape[1]
    for el in SEMIGROUP.elements:
        for n in range(num_samples):
            edge_label_list = []
            for b in range(num_branches):
                edge_label_list.append(paths[graphs[el][n][b]])
            p_cls = []
            for a_branch in edge_label_list:
                singleton_edge_labels = []
                for a_path in a_branch:
                    if check_sub_graph(a_path):
                        EL = [[[SEMIGROUP.str2int[edge[0]]] for edge in edges] for edges in a_path]
                        outs = []
                        for path in EL:
                            comp_out = compute_algebraic_closure_for_paths(path, SEMIGROUP.composition_table)
                            outs.append(comp_out)
                        singleton = intersect_sets(outs)
                        singleton_edge_labels.append(list(singleton))
                    else:
                        singleton_edge_labels.append([SEMIGROUP.str2int[a_path[0]]])
                p_cls.append(compute_algebraic_closure_for_paths(singleton_edge_labels, SEMIGROUP.composition_table))
            assert list(intersect_sets(p_cls))[0] == el, print(edge_label_list)