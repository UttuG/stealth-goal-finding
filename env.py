import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_grid_matrix(n=8):
    # Initialize the grid with zeros (empty cells)
    grid = np.zeros((n, n), dtype=int)
    return grid

def update_grid_matrix(grid_matrix, coordinates):
    # Update the grid matrix based on coordinates
    for i, (x, y) in enumerate(coordinates.get("guards", [])):
        grid_matrix[x, y] = 1  # Guard cell
    for x, y in coordinates.get("hero", []):
        grid_matrix[x, y] = 2  # Hero cell
    for x, y in coordinates.get("obstacles", []):
        grid_matrix[x, y] = 3  # Obstacle cell
    return grid_matrix

def extract_subgrid(grid_matrix, center, size):
    """
    Extract a subgrid directly using NumPy slicing.
    Returns a view when possible, which is tied to the original matrix.
    """
    n = grid_matrix.shape[0]
    half = size // 2
    
    # Calculate slice boundaries
    row_start = max(0, center[0] - half)
    row_end = min(n, center[0] + half + 1)
    col_start = max(0, center[1] - half)
    col_end = min(n, center[1] + half + 1)
    
    # Create the actual subgrid with proper padding
    # First create a grid of obstacles (all 3s)
    subgrid = np.full((size, size), 3, dtype=int)
    
    # Calculate where in the subgrid to place the actual values
    sub_row_start = half - (center[0] - row_start)
    sub_col_start = half - (center[1] - col_start)
    
    # Copy the relevant slice from the parent grid
    parent_slice = grid_matrix[row_start:row_end, col_start:col_end]
    subgrid[sub_row_start:sub_row_start+parent_slice.shape[0], 
            sub_col_start:sub_col_start+parent_slice.shape[1]] = parent_slice
    
    return subgrid

def plot_grid(grid_matrix, title="Main Grid", agent_indices=None):
    n_rows, n_cols = grid_matrix.shape
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    ax.set_title(title)
    
    # Draw grid lines
    for i in range(n_rows+1):
        ax.axhline(y=i, color='black', linestyle='-', linewidth=1)
    for j in range(n_cols+1):
        ax.axvline(x=j, color='black', linestyle='-', linewidth=1)
    
    # If agent_indices is None, create an empty dictionary
    if agent_indices is None:
        agent_indices = {}
    
    # Update cell colors based on grid_matrix
    for i in range(n_rows):
        for j in range(n_cols):
            if grid_matrix[i, j] == 0:  # Empty cell
                color = 'white'
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                ax.add_patch(rect)
            elif grid_matrix[i, j] == 1:  # Guard cell
                color = 'blue'
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                ax.add_patch(rect)
                
                # Add guard number if this position is in agent_indices
                for guard_num, pos in agent_indices.get("guards", {}).items():
                    if pos == (i, j):
                        ax.text(j + 0.5, i + 0.5, str(guard_num), 
                                color='white', ha='center', va='center', fontsize=12)
            elif grid_matrix[i, j] == 2:  # Hero cell
                color = 'green'
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                ax.add_patch(rect)
            elif grid_matrix[i, j] == 3:  # Obstacle cell
                color = 'black'
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                ax.add_patch(rect)
    
    plt.tight_layout()
    return fig, ax

class AgentSubgrid:
    def __init__(self, parent_grid, position, size, agent_type, agent_num):
        self.parent_grid = parent_grid
        self.position = position
        self.size = size
        self.agent_type = agent_type
        self.agent_num = agent_num
        
        # Create figure and axis for this agent's view
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.title = f"{agent_type} {agent_num} View ({size}x{size})"
        self.update()
    
    def update(self):
        """Update the subgrid view based on current parent grid state"""
        subgrid = extract_subgrid(self.parent_grid, self.position, self.size)
        
        # Clear previous plot
        self.ax.clear()
        
        # Configure axis
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(self.title)
        
        # Draw grid lines
        for i in range(self.size+1):
            self.ax.axhline(y=i, color='black', linestyle='-', linewidth=1)
            self.ax.axvline(x=i, color='black', linestyle='-', linewidth=1)
        
        # Plot the cells
        for i in range(self.size):
            for j in range(self.size):
                if subgrid[i, j] == 0:  # Empty cell
                    color = 'white'
                elif subgrid[i, j] == 1:  # Guard cell
                    color = 'blue'
                elif subgrid[i, j] == 2:  # Hero cell
                    color = 'green'
                elif subgrid[i, j] == 3:  # Obstacle cell
                    color = 'black'
                    
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
                self.ax.add_patch(rect)
        
        # Mark center with a red dot if it's not the current agent
        if subgrid[self.size//2, self.size//2] != (1 if self.agent_type == "Guard" else 2):
            self.ax.plot(self.size//2 + 0.5, self.size//2 + 0.5, 'ro')
        
        plt.figure(self.fig.number)
        self.fig.canvas.draw_idle()

def main():
    # Create the 8x8 grid
    n = 8
    grid_matrix = create_grid_matrix(n)
    
    # Define element locations with numbered guards
    coordinates = {
        "guards": [(0, 0), (3, 2)],  # Blue cells
        "hero": [(4, 3)],            # Green cell
        "obstacles": [(4, 5), (5, 6)] # Black cells
    }
    
    # Create a dictionary mapping guard numbers to positions
    guard_indices = {i+1: pos for i, pos in enumerate(coordinates["guards"])}
    agent_indices = {"guards": guard_indices}
    
    # Update the grid
    grid_matrix = update_grid_matrix(grid_matrix, coordinates)
    
    # Display the main grid with numbered guards
    main_fig, main_ax = plot_grid(grid_matrix, "Main Grid", agent_indices)
    plt.figure(main_fig.number)
    plt.show(block=False)
    
    # Create agent subgrids that are tied to the main grid
    agent_views = []
    
    # Hero views (5x5)
    for i, hero_pos in enumerate(coordinates["hero"]):
        hero_view = AgentSubgrid(grid_matrix, hero_pos, 5, "Hero", i+1)
        agent_views.append(hero_view)
    
    # Guard views (3x3)
    for i, guard_pos in enumerate(coordinates["guards"]):
        guard_view = AgentSubgrid(grid_matrix, guard_pos, 3, "Guard", i+1)
        agent_views.append(guard_view)
    
    # Display all plots
    plt.show()
    
    # Example of updating the grid and having all views update
    # Uncomment to test:
    # grid_matrix[2, 2] = 3  # Add a new obstacle
    # for agent_view in agent_views:
    #     agent_view.update()
    # main_fig.canvas.draw_idle()
    # plt.show()

# Run the main function
if __name__ == "__main__":
    main()
