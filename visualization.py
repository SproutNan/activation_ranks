import torch
import numpy as np
import matplotlib.pyplot as plt
from path import *
from rank import *
from ablation import *

def GetLayerNeuron(task_name: str):
    container = torch.load(times(task_name), map_location='cpu')
    return container.shape

def PlotSingleActivationTime(
    task_name: str, 
    layer_index: int, 
    step=5,
):
    container = torch.load(times(task_name), map_location='cpu')
    activations = container[layer_index]

    # sort by times
    sorted_indices = torch.argsort(activations, descending=True)
    sorted_activations = activations[sorted_indices]

    selected_indices = sorted_indices[::step]
    selected_activations = sorted_activations[::step]

    colors = plt.cm.viridis(torch.linspace(0, 1, len(sorted_activations)))

    plt.figure(figsize=(10, 2))
    plt.bar(range(len(selected_activations)), selected_activations, color=colors[selected_indices], width=1)
    plt.xlabel('Activation Rankings')
    plt.ylabel('Activation Times')
    plt.title(f'Activation Times for Layer {layer_index} in {task_name}')


def PlotTwoRankings(
    task_name1: str,
    task_name2: str,
    layer_index: int,
    ax="None",
):
    x_coordinates, y_coordinates = GetRankComparison(task_name1, task_name2, layer_index)

    colors = np.arange(x_coordinates.shape[0])

    if ax == "None":
        plt.figure(figsize=(5, 4))
        plt.scatter(x_coordinates, y_coordinates, c=colors, cmap='viridis', s=0.1)
        plt.title(f'Layer {layer_index+1} Rank Comparison')
        plt.xlabel('Rank in Matrix 1')
        plt.ylabel('Rank in Matrix 2')
        plt.colorbar(label='Neuron Index')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
    
    else:
        ax.scatter(x_coordinates, y_coordinates, c=colors, cmap='viridis', s=0.1)
        ax.set_title(f'Layer {layer_index+1} Rank Comparison')
        ax.set_xlabel('Rank in Matrix 1')
        ax.set_ylabel('Rank in Matrix 2')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def PlotAllLayers(
    task_name1: str,
    task_name2: str,
    total_layers: int,
    layers_per_row: int = 4
):
    rows = (total_layers + layers_per_row - 1) // layers_per_row
    # 增加图表宽度以留出空间给colorbar
    fig, axes = plt.subplots(rows, layers_per_row, figsize=(5 * layers_per_row + 1, 4 * rows), squeeze=False)

    for i in range(total_layers):
        row = i // layers_per_row
        col = i % layers_per_row
        PlotTwoRankings(task_name1, task_name2, i, axes[row, col])

    # 清理未使用的axes
    for ax in axes.flat[total_layers:]:
        ax.remove()

    # 调整布局，给colorbar留出空间
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)  # 调整此值以避免子图和colorbar重叠

    # 添加统一的colorbar
    _, n_neurons = GetLayerNeuron(task_name1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=n_neurons - 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # 调整colorbar的位置和大小
    fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label='Neuron Index')

def PlotTwoRankingsWithRect(
    task_name1: str,
    task_name2: str,
    layer_index: int,
    horizonal: tuple[int, int],
    vertical: tuple[int, int],
    cluster: list[int],
):
    x_coordinates, y_coordinates = GetRankComparison(task_name1, task_name2, layer_index)
    
    colors = np.arange(x_coordinates.shape[0])

    plt.figure(figsize=(5, 4))
    plt.scatter(x_coordinates, y_coordinates, c=colors, cmap='viridis', s=0.1)
    plt.title(f'Layer {layer_index+1} Rank Comparison')
    plt.xlabel('Rank in Matrix 1')
    plt.ylabel('Rank in Matrix 2')
    plt.colorbar(label='Neuron Index')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    left, right = horizonal
    upper, lower = vertical

    plt.axvline(left, color='red', linestyle='--')
    plt.axvline(right, color='red', linestyle='--')
    plt.axhline(upper, color='red', linestyle='--')
    plt.axhline(lower, color='red', linestyle='--')

    plt.scatter(x_coordinates[cluster], y_coordinates[cluster], c='red', s=1)
    plt.tight_layout()


def PlotTwoRankingsWithNeuron(
    task_name1: str,
    task_name2: str,
    layer_index: int,
    center_x: int,
    center_y: int,
    cluster: list[int],
):
    x_coordinates, y_coordinates = GetRankComparison(task_name1, task_name2, layer_index)
    
    colors = np.arange(x_coordinates.shape[0])

    plt.figure(figsize=(5, 4))
    plt.scatter(x_coordinates, y_coordinates, c=colors, cmap='viridis', s=0.1)
    plt.title(f'Layer {layer_index+1} Rank Comparison')
    plt.xlabel('Rank in Matrix 1')
    plt.ylabel('Rank in Matrix 2')
    plt.colorbar(label='Neuron Index')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.scatter(center_x, center_y, c='green', s=5)
    plt.scatter(x_coordinates[cluster], y_coordinates[cluster], c='red', s=1)
    plt.tight_layout()
