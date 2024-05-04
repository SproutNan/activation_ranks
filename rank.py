import torch
from path import *

def GetRankFromContainer(container):
    """```
    # Input
    tensor([[0.2708, 0.9249, 0.8448, 0.1566, 0.0166],
            [0.0508, 0.1186, 0.7267, 0.7806, 0.1324],
            [0.7497, 0.8353, 0.3633, 0.5634, 0.6607],
            [0.0253, 0.0276, 0.5766, 0.5355, 0.4895],
            [0.5507, 0.3378, 0.8436, 0.9839, 0.0962]])
    # Output
    tensor([[3., 5., 4., 2., 1.],
            [1., 2., 4., 5., 3.],
            [4., 5., 1., 2., 3.],
            [1., 2., 5., 4., 3.],
            [3., 2., 4., 5., 1.]])
    ```"""
    rank_k = torch.zeros_like(container)

    for i in range(container.shape[0]):
        row = container[i, :]
        _, rank = torch.sort(row)
        rank = rank.argsort() + 1
        rank_k[i, :] = rank

    return rank_k

def GetRankComparison(
    task_name1: str,
    task_name2: str,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return: Rank Comparison
    - x_coordinates = ranks_m1
    - y_coordinates = ranks_m2
    """
    container1 = torch.load(ranks(task_name1), map_location='cpu')
    container2 = torch.load(ranks(task_name2), map_location='cpu')

    ranks_m1 = container1[layer_index]
    ranks_m2 = container2[layer_index]

    ranks_m1 = ranks_m1.argsort().argsort()
    ranks_m2 = ranks_m2.argsort().argsort()

    return ranks_m1, ranks_m2

def GetRanksByLayer(
    task_name: str,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    container = torch.load(ranks(task_name), map_location='cpu')
    rank = container[layer_index, :]
    return rank