import torch
from path import *
from rank import *
import numpy as np

# 从激活排行 comparison 图上框选一部分神经元，返回所框选的神经元的编号
def ClusterByRect(
    task_name1: str,
    task_name2: str,
    layer_index: int,
    horizonal: tuple[int, int],
    vertical: tuple[int, int],
) -> list[int]:
    
    x_coordinates, y_coordinates = GetRankComparison(task_name1, task_name2, layer_index)
    
    left, right = horizonal
    upper, lower = vertical

    # 筛选出在框选区域内的神经元
    selected_neurons = []
    for i in range(len(x_coordinates)):
        if left <= x_coordinates[i] <= right and upper <= y_coordinates[i] <= lower:
            selected_neurons.append(i)

    return selected_neurons

# 从激活排行 comparison 图上选中一个神经元，自动计算其周围的神经元，返回所选神经元及其周围神经元的编号
def ClusterByNeuron(
    task_name1: str,
    task_name2: str,
    layer_index: int,
    center_x: int,
    center_y: int,
) -> list[int]:
    
    rank1 = GetRanksByLayer(task_name1, layer_index)
    rank2 = GetRanksByLayer(task_name2, layer_index)
    distances = torch.sqrt((rank1 - center_x).pow(2) + (rank2 - center_y).pow(2))

    closest_point = torch.argmin(distances)

    # 计算所有点离最近点的距离
    distances = torch.sqrt((rank1 - rank1[closest_point]).pow(2) + (rank2 - rank2[closest_point]).pow(2))

    # 计算所有点与最近点编号的差值
    indices = torch.arange(rank1.shape[0]).view(-1, 1)
    index_diffs = torch.abs(indices - indices[closest_point])

    # 根据上面两个距离计算所有点到最近点的综合距离，这里的综合距离是两个距离的加权和
    distances = 0.9 * distances + 0.06 * index_diffs.squeeze()

    # 距离越远，透明度越大
    alphas = 20 / (1 + distances)
    thershold = 0.1
    return torch.where(alphas > thershold)[0].tolist()