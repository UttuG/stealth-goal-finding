# grid_environment.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_grid_matrix(size):
    return np.zeros((size, size), dtype=int)

def update_grid_matrix(grid, coordinates):
    grid.fill(0)
    # Set obstacles (3)
    for x, y in coordinates.get("obstacles", []):
        grid[x, y] = 3
    # Set guards (1)
    for x, y in coordinates.get("guards", []):
        grid[x, y] = 1
    # Set heroes (2)
    for x, y in coordinates.get("heroes", []):
        grid[x, y] = 2
    # Set goal (4)
    for x, y in coordinates.get("goal", []):
        grid[x, y] = 4
    return grid

def extract_subgrid(grid, center, size):
    n = grid.shape[0]
    half = size // 2
    
    subgrid = np.full((size, size), 3, dtype=int)  # Start with obstacles
    
    # Calculate bounds
    min_row = max(0, center[0] - half)
    max_row = min(n, center[0] + half + 1)
    min_col = max(0, center[1] - half)
    max_col = min(n, center[1] + half + 1)
    
    # Calculate positions in subgrid
    sub_min_row = half - (center[0] - min_row)
    sub_min_col = half - (center[1] - min_col)
    
    # Insert actual values
    subgrid[sub_min_row:sub_min_row+(max_row-min_row), 
           sub_min_col:sub_min_col+(max_col-min_col)] = \
        grid[min_row:max_row, min_col:max_col]
    
    return subgrid

def create_subplot(ax, grid, title, size):
    """Helper to create consistent subplots"""
    ax.clear()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    
    # Draw grid lines
    for i in range(size+1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Create cells
    for i in range(size):
        for j in range(size):
            value = grid[i, j]
            color = 'white'
            if value == 1: color = 'blue'
            elif value == 2: color = 'green'
            elif value == 3: color = 'black'
            elif value == 4: color = 'yellow'
            
            rect = Rectangle((j, i), 1, 1, 
                           facecolor=color, 
                           edgecolor='black',
                           linewidth=0.5)
            ax.add_patch(rect)
    return ax
