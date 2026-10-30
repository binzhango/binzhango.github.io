import torch

def route(x, weight, k, bias=None):
    # x: [tokens, hidden], weight: [hidden, experts]
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected matrices")
    if x.shape[0] == 0 or not 1 <= k <= weight.shape[1]:
        raise ValueError("nonempty tokens and valid k required")
    probs = (x @ weight).float().softmax(-1)
    selection = probs if bias is None else probs + bias
    ids = selection.topk(k, dim=-1).indices
    gates = probs.gather(-1, ids)
    gates = gates / gates.sum(-1, keepdim=True)
    loads = torch.nn.functional.one_hot(ids, weight.shape[1]).float()
    fraction = loads.mean(dim=(0, 1))
    aux = weight.shape[1] * (fraction.detach() * probs.mean(0)).sum()
    return ids, gates, aux

x = torch.tensor([[1., 0.], [0., 1.]])
w = torch.tensor([[3., 1., 0.], [0., 1., 3.]])
ids, gates, aux = route(x, w, 2)
assert ids.tolist() == [[0, 1], [2, 1]]
assert torch.allclose(gates.sum(-1), torch.ones(2))
assert torch.isfinite(aux)
assert route(x, w, 3)[0].shape == (2, 3)
