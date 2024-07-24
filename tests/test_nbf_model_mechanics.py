from src.model_nbf_fb import NBFdistRModule, device, NBFdistR
from src.utils import get_doubly_stochasic_tensor
import torch
import pytest
from dataclasses import dataclass

@dataclass
class Batch:
    batch: torch.Tensor
    target_edge_index: torch.Tensor
    num_nodes: int

@pytest.fixture
def get_NBF_backward_module_facets():
    return NBFdistRModule(4, 4, facets=2)

@pytest.fixture
def get_NBF_backward_module_no_facets():
    return NBFdistRModule(4, 4, facets=1)

@pytest.fixture
def get_NBF_backward_model():
    hidden_dim=4
    facets=2
    batch = {'batch': torch.arange(6, device=device), 
            'target_edge_index': torch.tensor([[0, 1, 2, 5], 
                                               [1, 2, 3, 4]], device=device),
            'num_nodes': 6                                   
            }
    batch = Batch(**batch)
    return batch, NBFdistR(hidden_dim=hidden_dim, facets=facets, num_relations=18)

def check_A_correctness_for_base_relations(model):
    id = torch.eye(model.new_hidden_dim).to(device=device)
    for f in range(model.facets):
        assert torch.all(model.get_A()[f][..., 0, :] == id) # top rows
        assert torch.all(model.get_A()[f][..., 0, :] == id) # left cols


def test_A_comp_map_correctness_facets(get_NBF_backward_module_facets: NBFdistRModule):
    check_A_correctness_for_base_relations(get_NBF_backward_module_facets)


def test_A_comp_map_correctness(get_NBF_backward_module_no_facets: NBFdistRModule):
    check_A_correctness_for_base_relations(get_NBF_backward_module_no_facets)

def test_boundary_input_embeddings(get_NBF_backward_model):
    batch, model = get_NBF_backward_model
    out = model.make_boundary(batch)
    prob_sum = out.reshape(-1, model.facets, model.hidden_dim//model.facets).sum(axis=-1)
    # test 1: prob norm satisfaction
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum)), "source embeddings don't satisfy prob norm"
    # test 2: check that the head nodes are init'd as [[1,...,0], [1,...,0], ...]
    source_indices = batch.target_edge_index[0]
    target = torch.tensor(
            [
                [1., 0.,],
                [1., 0.,],
            ]
            ,device=device).reshape(-1)
    for ind in source_indices:
        assert torch.all(out[ind] == target), "source head embeddings are not 1-hot"
    # test 3: check that the intermediate nodes are init'd as max entropy prob dists
    for ind in [3, 4]:
        assert torch.allclose(out[ind], torch.tensor([0.5, 0.5, 0.5, 0.5], device=device)), "intermediate node embeddings are not max entropy prob dists"

def test_doubly_stochasic_generator():
    # 1-D case
    for _ in range(1000):
        x = get_doubly_stochasic_tensor(16)
        assert torch.allclose(x.sum(axis=-1), torch.tensor(1.))
    # 2-D case
    for _ in range(1000):
        x = get_doubly_stochasic_tensor((43, 43))
        assert torch.allclose(x.sum(axis=-1), torch.tensor(1.))
        assert torch.allclose(x.sum(axis=-2), torch.tensor(1.))
    # 3-D case
    for _ in range(1000):
        x = get_doubly_stochasic_tensor(12, 12, 12)
        assert torch.allclose(x.sum(axis=-1), torch.tensor(1.))
        assert torch.allclose(x.sum(axis=-2), torch.tensor(1.))
    # n-D case
    for n in range(4, 10):
        for _ in range(100):
            # small dim since x size is growing exponentially now
            x = get_doubly_stochasic_tensor(*tuple([4]*n))
            assert torch.allclose(x.sum(axis=-1), torch.tensor(1.))
            assert torch.allclose(x.sum(axis=-2), torch.tensor(1.))

def test_composition_correctness():
    # TODO already in the implementation code but maybe transfer out to here later... 
    pass
