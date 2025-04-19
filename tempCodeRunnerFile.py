if __name__ == "__main__":
    # For training without rendering:
    env = GridWorldEnv(grid_size=10, obstacle_count=10)
    # For visualization:
    # env = GridWorldEnv(grid_size=10, obstacle_count=10, render_mode="human")
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render #not required for vizualization now, only the render_mode parameter dictates the view
        time.sleep(0.1)
        if done:
            print(reward)
            obs = env.reset()
    env.close()
