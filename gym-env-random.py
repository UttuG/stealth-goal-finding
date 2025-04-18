#need to understand the matplotlib vizualization dynamic update and functions

import numpy as np
import gym 
from gym import spaces
import matplotlib.pyplot as plt
# from grid_environment import create_grid_matrix, update_grid_matrix, extract_subgrid, create_subplot
import time
from matplotlib.patches import Rectangle

#functions outside the env where imported from older grid-env file without gym
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


class GridWorldEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, grid_size=20, hero_count=1, hero_size=5, 
                 guard_count=2, guard_size=3, obstacle_count=10):
        super(GridWorldEnv, self).__init__()

        # Environment parameters
        self.hero_count = hero_count
        self.guard_count = guard_count
        self.guard_size = guard_size
        self.grid_size = grid_size
        self.hero_size = hero_size
        self.obstacle_count = obstacle_count

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0:up, 1:right, 2:down, 3:left
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(self.hero_size, self.hero_size),
            dtype=np.uint8
        )

        # Environment state
        self.grid_matrix = None
        self.coordinates = None
        self.explored = None
        self.episode_steps = 0
        self.max_episode_steps = 1000
        self.goal_reached = False

        # Visualization
        self.main_fig = None
        self.main_ax = None
        self.main_patches = None
        self.agent_views = []
        self.is_initialized = False

    def reset(self):
        self.episode_steps = 0
        self.goal_reached = False
        self.grid_matrix = create_grid_matrix(self.grid_size)
        self.explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Initialize coordinates
        self.coordinates = {
            "guards": [],
            "heroes": [],
            "obstacles": [],
            "goal": []
        }

        # Place heroes
        for _ in range(self.hero_count):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if pos not in self.coordinates["heroes"]:
                    self.coordinates["heroes"].append(pos)
                    break

        # Place guards
        for _ in range(self.guard_count):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if (pos not in self.coordinates["guards"] and 
                    pos not in self.coordinates["heroes"]):
                    self.coordinates["guards"].append(pos)
                    break

        # Place goal
        while True:
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if (pos not in self.coordinates["guards"] and 
                pos not in self.coordinates["heroes"]):
                self.coordinates["goal"].append(pos)
                break

        # Place obstacles
        for _ in range(self.obstacle_count):
            while True:
                pos = (np.random.randint(0, self.grid_size), 
                       np.random.randint(0, self.grid_size))
                if (pos not in self.coordinates["guards"] and 
                    pos not in self.coordinates["heroes"] and 
                    pos not in self.coordinates["obstacles"] and 
                    pos not in self.coordinates["goal"]):
                    self.coordinates["obstacles"].append(pos)
                    break

        # Update grid matrix
        self.grid_matrix = update_grid_matrix(self.grid_matrix, self.coordinates)
        
        # Initialize visualization
        if not self.is_initialized:
            self._init_visualization()
        else:
            self._update_all_views()

        # Return initial observation
        return self._get_observation()

    def step(self, action):
        self.episode_steps += 1
        reward = -0.1  # Time penalty
        done = False
        info = {}

        # Move main hero (assuming single hero for action space)
        old_pos = self.coordinates["heroes"][0]
        new_pos = self._move_agent(old_pos, action)
        
        # Update exploration
        self._update_exploration(new_pos)
        
        # Check goal reached
        if new_pos == self.coordinates["goal"][0]:
            reward += 10
            done = True
            self.goal_reached = True
        
        # Move guards
        self._move_guards()
        
        # Check guard detection
        if self._check_detection(new_pos):
            reward -= 10
            done = True
        
        # Update grid state
        self._update_grid_state(old_pos, new_pos)
        
        # Update visualization
        self._update_all_views()
        
        return self._get_observation(), reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            plt.pause(0.01)

    def close(self):
        if self.is_initialized:
            plt.close('all')
            self.is_initialized = False

    def _init_visualization(self):
        plt.ion()
        
        # Main grid
        self.main_fig, self.main_ax = plt.subplots(figsize=(8, 8))
        self._update_main_view()
        
        # Agent views
        self.agent_figs = []
        for idx, pos in enumerate(self.coordinates["heroes"]):
            fig, ax = plt.subplots(figsize=(4, 4))
            self._update_agent_view(ax, pos, self.hero_size, f"Hero {idx+1}")
            self.agent_figs.append(fig)
            
        for idx, pos in enumerate(self.coordinates["guards"]):
            fig, ax = plt.subplots(figsize=(4, 4))
            self._update_agent_view(ax, pos, self.guard_size, f"Guard {idx+1}")
            self.agent_figs.append(fig)
        
        plt.show(block=False)
        self.is_initialized = True

    def _update_main_view(self):
        self.main_ax.clear()
        self.main_ax.set_xticks([])
        self.main_ax.set_yticks([])
        self.main_ax.set_title("Main Grid")
        self.main_ax.set_xlim(0, self.grid_size)
        self.main_ax.set_ylim(0, self.grid_size)
        self.main_ax.set_aspect('equal', adjustable='box')

        for i in range(self.grid_size + 1):
            self.main_ax.axhline(i, color='black', linewidth=0.5)
            self.main_ax.axvline(i, color='black', linewidth=0.5)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.grid_matrix[i, j]
                color = 'white'
                if value == 1: color = 'blue'
                elif value == 2: color = 'green'
                elif value == 3: color = 'black'
                elif value == 4: color = 'yellow'
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                self.main_ax.add_patch(rect)
                # Draw guard number
                if value == 1:
                    for idx, pos in enumerate(self.coordinates["guards"]):
                        if pos == (i, j):
                            self.main_ax.text(j+0.5, i+0.5, str(idx+1), color='white',
                                            ha='center', va='center', fontsize=12, fontweight='bold')
                # Optionally, draw hero number
                elif value == 2:
                    for idx, pos in enumerate(self.coordinates["heroes"]):
                        if pos == (i, j):
                            self.main_ax.text(j+0.5, i+0.5, str(idx+1), color='white',
                                            ha='center', va='center', fontsize=12, fontweight='bold')

        self.main_ax.figure.canvas.draw_idle()


    def _update_agent_view(self, ax, pos, size, title):
        """Update individual agent view"""
        subgrid = extract_subgrid(self.grid_matrix, pos, size)
        create_subplot(ax, subgrid, title, size)
        ax.set_aspect('equal', adjustable='box')
        ax.figure.canvas.draw()

    def _create_agent_view(self, ax, pos, size):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        subgrid = extract_subgrid(self.grid_matrix, pos, size)
        patches = {}
        for i in range(size):
            for j in range(size):
                color = self._get_color_from_value(subgrid[i, j])
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                patches[(i, j)] = rect
        return patches

    def _update_all_views(self):
        """Update all visualizations"""
        self._update_main_view()
        
        # Update hero views
        for idx, pos in enumerate(self.coordinates["heroes"]):
            self._update_agent_view(
                self.agent_figs[idx].axes[0],
                pos,
                self.hero_size,
                f"Hero {idx+1}"
            )
        
        # Update guard views
        offset = len(self.coordinates["heroes"])
        for idx, pos in enumerate(self.coordinates["guards"]):
            self._update_agent_view(
                self.agent_figs[offset+idx].axes[0],
                pos,
                self.guard_size,
                f"Guard {idx+1}"
            )
        
        # Process GUI events
        [fig.canvas.flush_events() for fig in [self.main_fig] + self.agent_figs]

    def _get_color(self, pos):
        if pos in self.coordinates["goal"]:
            return "yellow"
        if pos in self.coordinates["heroes"]:
            return "green"
        if pos in self.coordinates["guards"]:
            return "blue"
        if pos in self.coordinates["obstacles"]:
            return "black"
        return "white"

    def _get_color_from_value(self, value):
        return {
            0: "white",
            1: "blue",
            2: "green",
            3: "black",
            4: "yellow"
        }[value]

    def _move_agent(self, old_pos, action):
        x, y = old_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        new_pos = (x + dx, y + dy)
        
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size and 
            new_pos not in self.coordinates["obstacles"]):
            return new_pos
        return old_pos

    def _update_exploration(self, pos):
        view = extract_subgrid(self.grid_matrix, pos, self.hero_size)
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                x = pos[0] - self.hero_size//2 + i
                y = pos[1] - self.hero_size//2 + j
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if not self.explored[x, y]:
                        self.explored[x, y] = True

    def _check_detection(self, hero_pos):
        """Check if hero is in any guard's view area"""
        for guard_pos in self.coordinates["guards"]:
            # Calculate view boundaries
            half_size = self.guard_size // 2
            min_x = guard_pos[0] - half_size
            max_x = guard_pos[0] + half_size + 1
            min_y = guard_pos[1] - half_size
            max_y = guard_pos[1] + half_size + 1
            
            # Check if hero is within this rectangle
            if (min_x <= hero_pos[0] < max_x and
                min_y <= hero_pos[1] < max_y):
                return True
        return False


    def _update_grid_state(self, old_pos, new_pos):
        # Update hero position
        self.grid_matrix[old_pos] = 0
        self.grid_matrix[new_pos] = 2
        self.coordinates["heroes"][0] = new_pos

    def _get_observation(self):
        hero_pos = self.coordinates["heroes"][0]
        return extract_subgrid(self.grid_matrix, hero_pos, self.hero_size)

    def _move_guards(self):
        for idx in range(len(self.coordinates["guards"])):
            old_pos = self.coordinates["guards"][idx]
            action = np.random.randint(4)
            new_pos = self._move_agent(old_pos, action)
            
            # Update guard position
            self.grid_matrix[old_pos] = 0
            self.grid_matrix[new_pos] = 1
            self.coordinates["guards"][idx] = new_pos

# Example usage
if __name__ == "__main__":
    env = GridWorldEnv(grid_size=10, obstacle_count=5)
    obs = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.1)
        
        if done:
            obs = env.reset()
    
    env.close()
