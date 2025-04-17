###: still doubt about it

import numpy as np
import gym 
from gym import spaces #for the observation and action space
import matplotlib.pyplot as plt
from grid_environment import {
    create_grid_matrix,
    update_grid_matrix,
    extract_subgrid
}
import time

class GridWorldEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self,grid_size,hero_count,hero_size,guard_count,guard_size,obstacle_count): #how to initialize the class
        super(GridWorldEnv,self).__init__() #using the init from the parent gridworldenv

        #values of the agent and grid size
        self.hero_count = hero_count
        self.guard_count = guard_count
        self.guard_size=guard_size #size of the guard FOV
        self.grid_size=grid_size
        self.hero_size=hero_size #size of the hero FOV
        self.obstacle_count = obstacle_count

        #action space
        self.action_space = spaces.Discrete(4) # 0:up, 1:right, 2:down, 3:left

        #observation space, defining what the hero sees
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(self.hero_size,self.hero_size),
            dtype=np.uint8
        )

        #initializing environment state
        self.grid_matrix = None #the parent grid matrix, can initialize the matrix here itself but making it in reset
        self.coordinates = None ###
        self.fig = None ###
        self.ax = None ###
        self.patches = None ###
        self.agent_views = [] #array of the FOV view for each agent (hero,guard)
        self.episode_steps = 0 #starting val of step
        self.max_episode_steps = 1000

        #initializing the visual components (whole ###)
        self.main_fig = None
        self.main_ax = None
        self.main_patches = None
        self.is_initialized = False 

    def reset(self): #the env after getting resetted
        self.episode_steps=0

        #initializing the parent matrix here so fresh run requires a reset
        self.grid_matrix = create_grid_matrix(self.grid_size)

        #defining the initial positions of the agents + goal, making the env versatile for multiple heroes and guards

        self.coordinates = {
            "guards":[],
            "heroes":[],
            "obstacles":[],
            "goal":[]
        }

        #random placement of initialized hero
        #heroes
        for _ in range(self.hero_count):
            while True:
                hero_x,hero_y=np.random.randint(0,self.grid_size-1),np.random.randint(0,self.grid_size-1)
                #checking if position empty:
                if (hero_x,hero_y) not in self.coordinates["heroes"]:
                    self.coordinates["heroes"].append((hero_x,hero_y))
                    break
        
        #guards
        for _ in range(self.guard_count):
            while True:
                guard_x,guard_y=np.random.randint(0,self.grid_size-1),np.random.randint(0,self.grid_size-1)
                #checking if position empty:
                if (guard_x,guard_y) not in self.coordinates["guards"] and (guard_x,guard_y) not in self.coordinates["heroes"]:
                    self.coordinates["guard"].append((guard_x,guard_y))
                    break

        #goal
        while True:
            goal_x,goal_y=np.random.randint(0,self.grid_size-1),np.random.randint(0,self.grid_size-1)
            if (goal_x,goal_y) not in self.coordinates["guards"] and (goal_x,goal_y) not in self.coordinates["heroes"]:
                self.coordinates["goal"].append((goal_x,goal_y))
                break
        
        #obstacles
        for _ in range(self.obstacle_count):
            while True:
                obs_x, obs_y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
                # Check if position is empty
                if ((obs_x, obs_y) not in self.coordinates["hero"] and 
                    (obs_x, obs_y) not in self.coordinates["guards"] and
                    (obs_x, obs_y) not in self.coordinates["obstacles"] and 
                    (obs_x, obs_y) not in self.coordinates["goal"]):
                    self.coordinates["obstacles"].append((obs_x, obs_y))
                    break
        
        #updating the parent grid matrix with the intialized positions
        self.grid_matrix=update_grid_matrix(self.grid_matrix,self.coordinates)

    def step(self,action): 
        # need to implement this with multiple agents and rewards based on total exploration 
        # (giving +ve small reward for total new tiles covered till now through FOV, so keep memory of visited tiles and small +ve reward with new tile seen)
        # also constant -ve timestep to boost faster exploration rather than no movement
        # till goal not seen (while being undetected), then focus on getting to the goal (big positive reward when path to goal and episode termination)
        # being undetected means not coming in any guard's FOV, if coming then very big -ve reward + episode termination
        #keep guard movement function seperate as before so that can be modified later

        #update visualization

        #get new observation
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _init_visualization(self):
        #need to modify the old visualization to accomodate mutliple heroes and goal too (yellow color)
        pass

    def _update_agent_view(self, agent):
        pass

    def _update_all_views(self):
        pass
    def _move_guards(self):
        pass


# Example usage for testing the environment
if __name__ == "__main__":
    # Create and test the environment
    env = GridWorldEnv(grid_size)  # Smaller grid for testing
    observation = env.reset()
    
    # Run a simple random agent
    for _ in range(20):
        action = env.action_space.sample()  # Random action
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)  # Slow down to visualize
        
        if done:
            print("Episode finished")
            observation = env.reset()
    
    env.close()



