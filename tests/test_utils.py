from src.utils import chain_edges, make_graph_edge_list

def test_chain_edges_correctness():
    for k in range(2, 100):
        for b in range(1, 100):
            edge_list, source, sink = chain_edges(k, b)
            assert len(edge_list) == k*b
            assert edge_list[-1][1]-edge_list[-1][0] == 1

def test_make_graph_edge_list_subloop_simple():
    G5 = [
        [[['a', 'b', 'c'], ['d', 'e', 'f']], 'y'],
        [[['a', 'b', 'c'], ['d', 'e', 'f']], 'y'],
    ]
    E5 = [(0,1), (1,2), (2,5), (0,3), (3,4), (4,5), (5,11),
        (0,6), (6,7), (7,10), (0,8), (8,9), (9,10), (10,11)]
    assert make_graph_edge_list(G5, depth=2) == E5

def test_make_graph_edge_list_subloop_translation():
    # cases with a sub loop at start, middle and end positions in the main loop
    G = [
        ['e', [['a', 'b', 'c'], ['d', 'e', 'f']], 'y'],
        ['u', [['v', 'w', 'x'], ['i', 'v', 'o']], 'l']
    ]
    E = [(0,1), (1,2), (2,3), (3,6), (1,4), (4,5), (5,6), (6,13), 
        (0,7), (7,8), (8,9), (9,12), (7,10), (10,11), (11,12), (12, 13)]
    G2 = [
        [[['a', 'b', 'c'], ['d', 'e', 'f']], 'e', 'y'],
        [[['v', 'w', 'x'], ['i', 'v', 'o']], 'u', 'l']
    ]
    E2 = [(0,1), (1,2), (2,5), (0,3), (3,4), (4,5), (5,6), (6,13), 
        (0,7), (7,8), (8,11), (0,9), (9,10), (10,11), (11,12), (12, 13)
        ]
    G3 = [
        ['e', 'y', [['a', 'b', 'c'], ['d', 'e', 'f']]],
        ['u', 'l', [['v', 'w', 'x'], ['i', 'v', 'o']]]
    ]
    E3 = [(0,1), (1,2), (2,3), (3,4), (4,13), (2,5), (5,6), (6,13), 
        (0,7), (7,8), (8,9), (9,10), (10,13), (8,11), (11,12), (12, 13)]
    
    assert make_graph_edge_list(G, depth=2) == E
    assert make_graph_edge_list(G2, depth=2) == E2
    assert make_graph_edge_list(G3, depth=2) == E3


def test_make_graph_edge_list_subloop_rcc8_example():

    graph =  [[[[['TPPI'], ['TPPI']], [['NTPPI'], ['NTPPI']]],
            ['TPP'],
            [[['DC'], ['PO']], [['EQ'], ['DC']]]],
            [[[['NTPP'], ['DC']], [['NTPP'], ['DC']]],
            ['EQ'],
            [[['TPPI'], ['TPPI']], [['TPPI'], ['TPP']]]],
            [[[['DC'], ['TPP']], [['EQ'], ['TPP']]],
            ['EQ'],
            [[['EQ'], ['EC']], [['EC'], ['TPPI']]]]]
    
    target = [(0,1),(1,3), (0,2), (2,3), (3,4), (4,5), (5,19), (4,6), (6,19),
            (0,7), (7,9), (0,8), (8,9), (9,10), (10,11), (11,19), (10,12), (12,19),
            (0,13), (13,15), (0,14), (14,15), (15,16), (16,17),(17,19), (16, 18), (18,19),
            ]
    
    assert make_graph_edge_list(graph) == target

def test_make_graph_edge_list_subloop_varied():
    G = [
        ['e', 'v', [['a', 'b', 'c'], ['d', 'e', 'f']], 'y', [['a', 'b'], ['d', 'e']]],
        ['u', 'v', [['v', 'w', 'x'], ['i', 'v', 'o']], 'l', [['a', 'b'], ['d', 'e']]]
    ]
    E = [(0,1), (1,2), (2,3), (3,4), (4,7), (2,5), (5,6), (6,7), (7,8), (8,9), (9,21),(8, 10), (10,21),
        (0,11), (11,12), (12,13), (13,14), (14,17), (12,15), (15,16), (16,17), (17,18), (18,19),
        (19,21), (18,20), (20,21)]
    
    assert make_graph_edge_list(G, depth=2) == E