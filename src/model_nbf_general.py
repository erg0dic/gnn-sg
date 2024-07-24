from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from typing import Union

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Logits:
    pass
class Probas:
    pass

def entropy(probas, axis=0, eps=1e-8):
    "add a small eps as I want to avoid the log(0)=-inf val"
    out =  -1*(probas*torch.log(probas+eps)).sum(axis=axis)
    return out

def stable_norm_denominator(vector_batch, vec_dim=-1):
    # unused so far... but smart money bets it will be useful later
    norms = torch.norm(vector_batch, dim=vec_dim)
    norms[norms==0] = 1
    return norms.unsqueeze(-1)

def compute_sim(input, relation_in_proto_basis, type='cosine', einsum_str: str = None):
    if einsum_str is None:
        einsum_str = "nh, ph -> np"
    if type == 'cosine':
        # normalize vectors without hitting the zeros
        input = normalize(input)
        relation_in_proto_basis = normalize(relation_in_proto_basis)
        sims = torch.einsum(einsum_str, input, relation_in_proto_basis)
        return sims
    elif type=='euclidean':
        EPS=1e-8
        x = relation_in_proto_basis-input.unsqueeze(1)
        dist = torch.pow((x*x).sum(axis=-1), 0.5)
        return torch.log((dist+1)/(dist+EPS))
    else:
        raise NotImplementedError(f"Distance type {type} has not been implemented.")
    
def make_probas(x: torch.Tensor):
    # hacky? is it even worth testing or should I just use softmax on logprobs?
    return torch.abs(x) / x.sum(dim=-1).unsqueeze(-1)

def sample_gumbel(shape, eps=1e-10):
    # from G(0,1)
    # stablize cf. https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e
    U = torch.rand(shape, device=device)
    gumbles = -torch.log(-torch.log(U + eps) + eps)
    if torch.isnan(gumbles).any() or torch.isinf(gumbles).any():
        gumbles = sample_gumbel(shape, eps=eps)
    return gumbles

def gumbel_softmax_sample(logits: Logits, 
                          temperature=.7, 
                          dim=-1, 
                          eval_mode=False, 
                          just_discretize=False, 
                          ys_are_probas=False) -> Union[Logits, Probas]:
    if eval_mode:
        if ys_are_probas:
            return torch.softmax(logits, dim=-1)
        else:
            return logits
    # from Gumbel Softmax
    y = logits if just_discretize else logits + sample_gumbel(logits.shape)
    SO = torch.softmax if ys_are_probas else torch.log_softmax
    out =  SO(y / temperature, dim=dim)

    return out

class NBFCluttr(MessagePassing):
    def __init__(self, num_relations, hidden_dim, num_prototypes, 
                 dist='cosine', fix_ys=False):
        super().__init__()
        # mapping to multiple prototypes basis from 1 relation
        # Note the num_protos p can in fact be > num_relations r 
        # for the y(i,r) basis vectors of the message passing function
        if fix_ys:
            self.multi_embedding = (torch.rand(size=(num_relations, num_prototypes, hidden_dim), device=device))
        else:
            self.multi_embedding = nn.Parameter(torch.rand(size=(num_relations, num_prototypes, hidden_dim), device=device))
        # for the probas
        self.proto_embedding = nn.Parameter(torch.rand(size=(num_prototypes, hidden_dim)))
        self.dist = dist
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU() 
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)


    def forward(self, input, edge_index, edge_type, return_probas=False):
        return self.propagate(edge_index=edge_index, input=input, edge_type=edge_type,
                              return_probas=return_probas, num_nodes=input.shape[0])
    
    def look_up_rel_proto_vectors(self, batch_edge_indices):
        rel_basis = self.multi_embedding.index_select(0, batch_edge_indices)
        return rel_basis
    
    def message(self, input_j, edge_type):
        'the direction of propagation by default is o --> o i.e. left to right source-to-target.'
        # `input_j` just copies over the node vectors for each head in `edge_index`
        # so thinking we have a 2-D tensor (num_nodes, hidden_dim)
        # this will need to be the output shape too!
        rel_proto_basis = self.look_up_rel_proto_vectors(edge_type)
        proto_probas = self.project_nodes_onto_proto_simplex(input_j, self.proto_embedding, 
                                                             dist_type=self.dist)
        # compute entropy of each prototype vector in the proto basis
        out = torch.einsum("np, nph -> nh", proto_probas, rel_proto_basis)
        return out, proto_probas
        
    def aggregate(self, input, index, num_nodes):
        input_j, proto_proba = input # proto_proba is (protos, num_nodes)
        # add self loops to index
        proto_entropy = entropy(proto_proba, axis=1) 
        # the rest need to be scatter operations since they are over the node dimension
        entropic_attention_coeffs = scatter_softmax(-proto_entropy, index)
        out = torch.einsum('n, nh-> nh', entropic_attention_coeffs, input_j)
        out = scatter_add(out, index, dim=0, dim_size=num_nodes) # in the node dimension
        
        return out, proto_proba
    
    def project_nodes_onto_proto_simplex(self, inp, proto_basis, 
                                        dist_type='cosine'):
        # take a dot product between an `inp` node embedding vector 
        # and each prototypical relation vector and then softmax over the proto dim
        # if len(proto_basis.shape) == 2:
        #     proto_basis = proto_basis.expand(inp.shape[0], *proto_basis.shape)
        proto_input_overlap = compute_sim(inp, proto_basis, 
                                          type=dist_type)
        # This is scatter-invariant Because we have essentially copied 
        # over all prototypes for each edge input and taken a softmax 
        # over the prototypical basis.
        proto_proba = torch.softmax(proto_input_overlap, dim=1)
        return proto_proba

    def update(self, update, input, return_probas=False, mix_inp_out=False):
        # node update as a function of old states (input) [THIS IS THE INPUT TO `forward`] and this layer output (update)
        # print(update.shape, input.shape)
        new_out, proto_proba = update
        if mix_inp_out:
            output = self.linear(torch.cat([input, new_out], dim=-1))
            if self.layer_norm:
                output = self.layer_norm(output)
            if self.activation:
                output = self.activation(output)
        
        if return_probas:
            return new_out, proto_proba
        else:
            return new_out
        

class NBF_base(nn.Module):
    def __init__(self):
        super().__init__()

    def post_init(self):
        raise NotImplementedError
    
    @staticmethod
    def get_mlp_classifier(num_mlp_layer, hidden_dim, out_dim, middle_dim=None):
        nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            if middle_dim:
                if i==0:
                    mlp.append(nn.Linear(hidden_dim, middle_dim))
                else:
                    mlp.append(nn.Linear(middle_dim, middle_dim))
            else:
                mlp.append(nn.Linear(hidden_dim, hidden_dim))
            mlp.append(nn.Tanh())
        mlp.append(nn.Linear(middle_dim if middle_dim else hidden_dim, out_dim))
        return nn.Sequential(*mlp)

    def make_boundary(self, batch, source_embedding):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError
    
    def apply_classifier(self, hidden, tail_indices):
        raise NotImplementedError

        
class NBF(NBF_base):
    def __init__(self, hidden_dim, residual=False, num_mlp_layer=2, 
                 num_layers=10, num_relations=18, 
                 shared=False, dist='cosine', 
                 use_mlp_classifier=False, fix_ys=False, eval_mode=False):
        super().__init__()

        self.shared = shared
        self.use_mlp_classifier = use_mlp_classifier
        self.source_embedding = nn.Parameter(torch.rand(hidden_dim), requires_grad=True)
        self.hidden_dim = hidden_dim
        self.BF_layers = nn.ModuleList()
        self.residual = residual
        self.dist = dist
        self.eval_mode = eval_mode
        for i in range(num_layers-1):
            num_prototypes = num_relations if not self.use_mlp_classifier else 20
            self.BF_layers.append(NBFCluttr(num_relations, hidden_dim, 
                                            num_prototypes=num_prototypes, 
                                            dist=dist, fix_ys=fix_ys))
        
        if self.use_mlp_classifier:
            self.mlp = self.get_mlp_classifier(num_mlp_layer, hidden_dim, num_relations)

    def make_boundary(self, batch, source_embedding):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError
    
    def apply_classifier(self, hidden, tail_indices):
        out_embeddings = hidden[tail_indices]
        if self.use_mlp_classifier:
            logits = self.mlp(out_embeddings)
            return logits
        else:
            gcn_layer: NBFCluttr = self.BF_layers[0] if self.shared else self.BF_layers[-1]
            proto_embedding = gcn_layer.proto_embedding 
            sims = compute_sim(out_embeddings, proto_embedding, type=self.dist, einsum_str="nh, ph -> np")
            out_probas = gumbel_softmax_sample(sims, temperature=0.7, 
                                               dim=-1, eval_mode=self.eval_mode) # note this is not a prob but a logprob
            # logits = torch.log((out_probas)/(1-out_probas))
            # return logits
            return out_probas
