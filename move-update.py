# movement_loop.py

# Import the environment module
from grid_environment import (
    create_grid_matrix, 
    update_grid_matrix,
    extract_subgrid
)
import matplotlib.pyplot as plt
import numpy as np
import time

def update_agent_view(ax, patches, grid_matrix, agent_pos, view_size):
    """Update any agent's view based on current grid state and agent position"""
    subgrid = extract_subgrid(grid_matrix, agent_pos, view_size)
    
    for i in range(view_size):
        for j in range(view_size):
            color = 'white'  # Default color
            if subgrid[i, j] == 1:
                color = 'blue'  # Guard
            elif subgrid[i, j] == 2:
                color = 'green'  # Hero
            elif subgrid[i, j] == 3:
                color = 'black'  # Obstacle
            
            patches[(i, j)].set_facecolor(color)
    
    ax.figure.canvas.draw_idle()
    ax.figure.canvas.flush_events()

def update_main_grid(grid_matrix, main_patches):
    """Update the main grid view based on current grid state"""
    n = grid_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            color = 'white'  # Default color
            
            if grid_matrix[i, j] == 1:
                color = 'blue'  # Guard
            elif grid_matrix[i, j] == 2:
                color = 'green'  # Hero
            elif grid_matrix[i, j] == 3:
                color = 'black'  # Obstacle
                
            main_patches[(i, j)].set_facecolor(color)

def simple_movement_loop():
    # Create the grid
    n = 20  # Using larger grid 
    grid_matrix = create_grid_matrix(n)
    
    # Initialize with positions
    coordinates = {
        "guards": [(10, 9), (16, 3)],  # Blue guards
        "hero": [(5, 14)],             # Green hero
        "obstacles": [(0, 0), (19, 0)]  # Black obstacles in corners
    }
    
    # Update the grid matrix
    grid_matrix = update_grid_matrix(grid_matrix, coordinates)
    
    # INITIALIZE ALL PLOTS ONCE
    plt.ion()  # Turn on interactive mode
    
    # Main grid initial setup
    main_fig, main_ax = plt.subplots(figsize=(8, 8))
    main_ax.set_xlim(0, n)
    main_ax.set_ylim(0, n)
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    main_ax.set_title("Main Grid")
    
    # Draw grid lines once
    for i in range(n+1):
        main_ax.axhline(y=i, color='black', linestyle='-', linewidth=1)
        main_ax.axvline(x=i, color='black', linestyle='-', linewidth=1)
    
    # Initialize rectangles for the main grid
    main_patches = {}
    for i in range(n):
        for j in range(n):
            rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='none')
            main_ax.add_patch(rect)
            main_patches[(i, j)] = rect
    
    # Set initial colors based on grid_matrix
    update_main_grid(grid_matrix, main_patches)
    
    # Create and store all agent views
    agent_views = []
    
    # Hero views setup
    hero_size = 5
    for h_idx, hero_pos in enumerate(coordinates["hero"]):
        h_fig, h_ax = plt.subplots(figsize=(5, 5))
        h_patches = {}
        
        h_ax.set_xlim(0, hero_size)
        h_ax.set_ylim(0, hero_size)
        h_ax.set_xticks([])
        h_ax.set_yticks([])
        h_ax.set_title(f"Hero {h_idx+1} View ({hero_size}x{hero_size})")
        
        # Draw hero grid lines
        for i in range(hero_size+1):
            h_ax.axhline(y=i, color='black', linestyle='-', linewidth=1)
            h_ax.axvline(x=i, color='black', linestyle='-', linewidth=1)
        
        # Initialize hero subgrid patches
        for i in range(hero_size):
            for j in range(hero_size):
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='none')
                h_ax.add_patch(rect)
                h_patches[(i, j)] = rect
        
        # Initial update
        update_agent_view(h_ax, h_patches, grid_matrix, hero_pos, hero_size)
        
        agent_views.append({
            "type": "hero",
            "index": h_idx,
            "position": hero_pos,
            "ax": h_ax,
            "patches": h_patches,
            "size": hero_size
        })
    
    # Guard views setup
    guard_size = 3
    for g_idx, guard_pos in enumerate(coordinates["guards"]):
        g_fig, g_ax = plt.subplots(figsize=(5, 5))
        g_patches = {}
        
        g_ax.set_xlim(0, guard_size)
        g_ax.set_ylim(0, guard_size)
        g_ax.set_xticks([])
        g_ax.set_yticks([])
        g_ax.set_title(f"Guard {g_idx+1} View ({guard_size}x{guard_size})")
        
        # Draw guard grid lines
        for i in range(guard_size+1):
            g_ax.axhline(y=i, color='black', linestyle='-', linewidth=1)
            g_ax.axvline(x=i, color='black', linestyle='-', linewidth=1)
        
        # Initialize guard subgrid patches
        for i in range(guard_size):
            for j in range(guard_size):
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='none')
                g_ax.add_patch(rect)
                g_patches[(i, j)] = rect
        
        # Initial update
        update_agent_view(g_ax, g_patches, grid_matrix, guard_pos, guard_size)
        
        agent_views.append({
            "type": "guard",
            "index": g_idx,
            "position": guard_pos,
            "ax": g_ax,
            "patches": g_patches,
            "size": guard_size
        })
    
    # Display all plots
    plt.show(block=False)
    
    # Movement loop
    try:
        moving_right = True
        
        while True:
            # Choose which guard to move (first guard in this example)
            g_idx = 0
            current_pos = coordinates["guards"][g_idx]
            x, y = current_pos
            
            # Clear current position in grid matrix
            grid_matrix[x, y] = 0
            
            # Move left or right
            if moving_right:
                new_y = y + 1
                if new_y >= n or grid_matrix[x, new_y] != 0:  # Check boundary or collision
                    new_y = y - 1
                    moving_right = False
            else:
                new_y = y - 1
                if new_y < 0 or grid_matrix[x, new_y] != 0:  # Check boundary or collision
                    new_y = y + 1
                    moving_right = True
            
            # Set new position in grid matrix
            grid_matrix[x, new_y] = 1
            
            # Update agent position in coordinates dictionary
            new_pos = (x, new_y)
            coordinates["guards"][g_idx] = new_pos
            
            # Update the main grid view
            update_main_grid(grid_matrix, main_patches)
            main_fig.canvas.draw_idle()
            main_fig.canvas.flush_events()
            
            # Update all agent views
            for agent in agent_views:
                if agent["type"] == "guard" and agent["index"] == g_idx:
                    # Update this agent's position in our tracking dictionary
                    agent["position"] = new_pos
                
                # Update the agent's view (regardless of whether it moved)
                update_agent_view(
                    agent["ax"], 
                    agent["patches"], 
                    grid_matrix, 
                    agent["position"], 
                    agent["size"]
                )
            
            # Small pause to make the movement visible
            plt.pause(0.5)
    
    except KeyboardInterrupt:
        print("Simulation stopped (Ctrl+C pressed)")
    
    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Keep windows open after the loop ends

if __name__ == "__main__":
    simple_movement_loop()
