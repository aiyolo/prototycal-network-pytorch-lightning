import torch

def euclidean_metric(a, b):
    return torch.cdist(a, b, p=2)