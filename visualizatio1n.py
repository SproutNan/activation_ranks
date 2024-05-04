import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

def draw_given_center(rank1, rank2, layer: int, point: tuple[int, int]):
    # point 是给定的中心，找到离中心最近的点
    rank1 = rank1[layer, :]
    rank2 = rank2[layer, :]

    rank1 = torch.max(rank1) - rank1
    rank2 = torch.max(rank2) - rank2

    x_diff = rank1 - point[0]
    y_diff = rank2 - point[1]
    distances = torch.sqrt(x_diff.pow(2) + y_diff.pow(2))

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
    alphas[alphas > thershold] = 1
    alphas[alphas <= thershold] = 0

    # 颜色是标号，透明度由距离决定
    colors = np.arange(rank1.shape[0])
    
    plt.figure(figsize=(5, 4))
    plt.colorbar(cm.ScalarMappable(cmap=cm.viridis), label='Neuron Index')
    plt.scatter(rank1.numpy(), rank2.numpy(), c=colors, s=0.1, alpha=alphas)
    plt.scatter(rank1[closest_point].item(), rank2[closest_point].item(), color='red', s=5)
    plt.title('Cluster Extraction')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    ret = alphas > thershold
    # 返回所有透明度大于阈值的点的编号
    return torch.where(ret)[0]

    # TODO: 现在这个提取算法不错，但是没法区分很乱的情况，需要引入周围熵进行进一步优化