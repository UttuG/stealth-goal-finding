###still need to understand the plotting and dynamic update

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

# --- Grid utility functions ---

def create_grid_matrix(size):
    return np.zeros((size, size), dtype=int)

def update_grid_matrix(grid, coordinates):
    grid.fill(0)
    for x, y in coordinates.get("obstacles", []):
        grid[x, y] = 3
    for x, y in coordinates.get("guards", []):
        grid[x, y] = 1
    for x, y in coordinates.get("heroes", []):
        grid[x, y] = 2
    for x, y in coordinates.get("goal", []):
        grid[x, y] = 4
    return grid

def extract_subgrid(grid, center, size):
    n = grid.shape[0]
    half = size // 2
    subgrid = np.full((size, size), 3, dtype=int)  # Start with obstacles
    min_row = max(0, center[0] - half)
    max_row = min(n, center[0] + half + 1)
    min_col = max(0, center[1] - half)
    max_col = min(n, center[1] + half + 1)
    sub_min_row = half - (center[0] - min_row)
    sub_min_col = half - (center[1] - min_col)
    subgrid[sub_min_row:sub_min_row+(max_row-min_row),
            sub_min_col:sub_min_col+(max_col-min_col)] = \
        grid[min_row:max_row, min_col:max_col]
    return subgrid

def create_subplot(ax, grid, title, size):
    ax.clear()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    for i in range(size+1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    for i in range(size):
        for j in range(size):
            value = grid[i, j]
            color = 'white'
            if value == 1: color = 'blue'
            elif value == 2: color = 'green'
            elif value == 3: color = 'black'
            elif value == 4: color = 'yellow'
            rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    return ax

# --- Main environment class ---

class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, grid_size=20, hero_count=1, hero_size=5,
                 guard_count=2, guard_size=3, obstacle_count=10,
                 render_mode=None):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.hero_count = hero_count
        self.guard_count = guard_count
        self.guard_size = guard_size #guard's observation grid size
        self.hero_size = hero_size
        self.obstacle_count = obstacle_count
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)  # 0:up, 1:right, 2:down, 3:left

        low = np.concatenate([
            np.zeros(self.hero_size * self.hero_size),              # subgrid: 0-4
            np.zeros(1),                                            # exploration_count: 0-1
            np.zeros(self.grid_size * self.grid_size)               # explored: 0-1
        ])

        high = np.concatenate([
            np.full(self.hero_size * self.hero_size, 4),            # subgrid: 0-4
            np.ones(1),                                             # exploration_count: 0-1
            np.ones(self.grid_size * self.grid_size)                # explored: 0-1
        ])

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        #note: the subgrid and explored values should be be integer but during sampling float32 values can be generated
        #a workaround for this can be to use space dict but then sb3 can cause issues

        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=4,  # or 1 for explored; adjust as needed
        #     shape=(self.hero_size * self.hero_size + 1 + self.grid_size * self.grid_size, ),
        #     dtype=np.float32  # Use float32 for everything for simplicity
        # )

        self.grid_matrix = None
        self.coordinates = None
        self.explored = None #2-D array to track explored tiles
        self.episode_steps = 0
        self.reward = 0
        self.max_episode_steps = 1000
        # coefficient for exploration bonus (equal in magnitude to your time penalty)
        self.exploration_bonus = 0.1  
        # flag to stop exploration bonus once the goal enters the subgrid 
        self.goal_seen = False
        self.goal_reached = False
        self.explored_tiles = set() # Set to track explored tiles

        # Visualization
        self.main_fig = None
        self.main_ax = None
        self.agent_figs = []
        self.is_initialized = False

    def reset(self, *, seed=None, options=None):
        self.episode_steps = 0
        self.reward = 0
        self.goal_reached = False
        self.goal_seen = False
        
        self.grid_matrix = create_grid_matrix(self.grid_size)
        #this is to record how much the hero has explored(seen) in the total grid
        self.explored = np.zeros((self.grid_size, self.grid_size), dtype=int) #kinda redundant tbh as set is also being used, only purpose is for observation for model and viz
        self.explored_tiles = set()
        ##
        
        self.coordinates = {
            "heroes": [],
            "obstacles": [],
            "goal": [],
            "guards": []
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

        self.grid_matrix = update_grid_matrix(self.grid_matrix, self.coordinates) #updatng the matrix with the generated coordinates

        if self.render_mode == "human" and not self.is_initialized:
            self._init_visualization()
        elif self.render_mode == "human":
            self._update_all_views()

        return self._get_observation()

    def step(self, action):
        self.episode_steps += 1
        self.reward += -0.1  # Time penalty for each timestep
        done = False
        info = {} #just part of the gym env, no use to me personally

        # Move main hero (assuming single hero for action space)
        old_pos = self.coordinates["heroes"][0]
        new_pos = self._move_agent(old_pos, action)
        
        prev_count = len(self.explored_tiles)
        self._update_exploration(new_pos)
        newly_seen = len(self.explored_tiles) - prev_count

        if not self.goal_seen:
            local = extract_subgrid(self.grid_matrix, new_pos, self.hero_size)
            self.reward += self.exploration_bonus * newly_seen
            if 4 in local:
                self.goal_seen = True
        

        # Check goal reached
        if new_pos == self.coordinates["goal"][0]:
            self.reward += 1000
            done = True
            self.goal_reached = True

        # Move guards
        self._move_guards()

        # Check guard detection
        if self._check_detection(new_pos):
            self.reward -= 1000
            done = True

        # Update grid state
        self._update_grid_state(old_pos, new_pos)

        if self.render_mode == "human":
            self._update_all_views()

        return self._get_observation(), self.reward, done, info

    def render(self):
        if self.render_mode == "human":
            plt.pause(0.01)

    def close(self):
        if self.is_initialized:
            plt.close('all')
            self.is_initialized = False

    # --- Visualization Methods ---

    def _init_visualization(self):
        plt.ion()
        self.main_fig, self.main_ax = plt.subplots(figsize=(8, 8))
        self._update_main_view()
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
                if value == 1:
                    for idx, pos in enumerate(self.coordinates["guards"]):
                        if pos == (i, j):
                            self.main_ax.text(j+0.5, i+0.5, str(idx+1), color='white',
                                              ha='center', va='center', fontsize=12, fontweight='bold')
                elif value == 2:
                    for idx, pos in enumerate(self.coordinates["heroes"]):
                        if pos == (i, j):
                            self.main_ax.text(j+0.5, i+0.5, str(idx+1), color='white',
                                              ha='center', va='center', fontsize=12, fontweight='bold')
        self.main_ax.figure.canvas.draw_idle()

    def _update_agent_view(self, ax, pos, size, title):
        subgrid = extract_subgrid(self.grid_matrix, pos, size)
        create_subplot(ax, subgrid, title, size)
        ax.set_aspect('equal', adjustable='box')
        ax.figure.canvas.draw()

    def _update_all_views(self):
        self._update_main_view()
        for idx, pos in enumerate(self.coordinates["heroes"]):
            self._update_agent_view(
                self.agent_figs[idx].axes[0],
                pos,
                self.hero_size,
                f"Hero {idx+1}"
            )
        offset = len(self.coordinates["heroes"])
        for idx, pos in enumerate(self.coordinates["guards"]):
            self._update_agent_view(
                self.agent_figs[offset+idx].axes[0],
                pos,
                self.guard_size,
                f"Guard {idx+1}"
            )
        [fig.canvas.flush_events() for fig in [self.main_fig] + self.agent_figs]

    # --- Grid and Agent Logic ---

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
                        self.explored[x, y] = 1
                        self.explored_tiles.add((x, y))

    def _check_detection(self, hero_pos):
        for guard_pos in self.coordinates["guards"]:
            half_size = self.guard_size // 2
            min_x = guard_pos[0] - half_size
            max_x = guard_pos[0] + half_size + 1
            min_y = guard_pos[1] - half_size
            max_y = guard_pos[1] + half_size + 1
            if (min_x <= hero_pos[0] < max_x and
                min_y <= hero_pos[1] < max_y):
                return True
        return False

    def _update_grid_state(self, old_pos, new_pos):
        self.grid_matrix[old_pos] = 0
        self.grid_matrix[new_pos] = 2
        self.coordinates["heroes"][0] = new_pos

    def _get_observation(self):
        # hero_pos = self.coordinates["heroes"][0]
        # subgrid = extract_subgrid(self.grid_matrix, hero_pos, self.hero_size)
        # exploration_count = len(self.explored_tiles) / (self.grid_size ** 2) # Normalize exploration count to [0, 1] 

        # return [subgrid,np.array([exploration_count], dtype=np.float32), self.explored] #can modify this to include more information if needed (not using this as sb3 needs numpy arrays)
        
        #repetition due to change in numpy array
        hero_pos = self.coordinates["heroes"][0]
        subgrid = extract_subgrid(self.grid_matrix, hero_pos, self.hero_size).astype(np.float32).flatten()
        exploration_count = np.array([len(self.explored_tiles) / (self.grid_size ** 2)], dtype=np.float32)
        explored = self.explored.astype(np.float32).flatten()
        obs = np.concatenate([subgrid, exploration_count, explored])
        return obs

    def _move_guards(self):
        # for idx in range(len(self.coordinates["guards"])):
        #     old_pos = self.coordinates["guards"][idx]
        #     action = np.random.randint(4)
        #     new_pos = self._move_agent(old_pos, action)
        #     self.grid_matrix[old_pos] = 0
        #     self.grid_matrix[new_pos] = 1
        #     self.coordinates["guards"][idx] = new_pos
        pass  # Placeholder for guard movement logic as training for static guards
        # This can be implemented based on specific requirements.

# --- Example usage ---

if __name__ == "__main__":
    # For training without rendering:
    # env = GridWorldEnv(grid_size=10, obstacle_count=40)
    # For visualization:
    env = GridWorldEnv(grid_size=10, obstacle_count=10, render_mode="human")
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render #not required for vizualization now, only the render_mode parameter dictates the view
        time.sleep(0.1)
        if done:
            print(reward, obs[2])
            obs = env.reset()
    # print(env.observation_space.sample())
    env.close()
