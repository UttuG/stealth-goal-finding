import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gym_env_random import GridWorldEnv


class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback that logs episode rewards and current timestep.
    Works with vectorized envs by checking 'episode' in each info dict.
    """
    def __init__(self):
        super(EpisodeLoggerCallback, self).__init__()
        self.episode_num = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_num += 1
                ep_reward = info['episode']['r']
                current_timestep = self.num_timesteps
                print(f"Episode {self.episode_num} done at timestep {current_timestep}, reward: {ep_reward}")
        return True


def make_env_fn(obstacles, guards, rank):
    """
    Return a function that creates and returns a Monitor-wrapped GridWorldEnv.
    """
    def _init():
        env = GridWorldEnv(
            grid_size=20,
            obstacle_count=obstacles,
            guard_count=guards,
            hero_size=5,
            guard_size=3,
            render_mode=None
        )
        return Monitor(env)
    return _init


if __name__ == "__main__":
    total_timesteps = 200_000
    phase_timesteps = 40_000
    n_phases = total_timesteps // phase_timesteps
    n_envs = 6  # number of parallel environments (matched to CPU cores)

    model = None
    callback = EpisodeLoggerCallback()

    for phase in range(n_phases):
        obstacles = 5 + 5 * phase
        guards = phase
        # Create vectorized environments
        env_fns = [make_env_fn(obstacles, guards, i) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)

        if model is None:
            print(f"Starting phase {phase+1}: obstacles={obstacles}, guards={guards}")
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1
            )
        else:
            print(f"Switching to phase {phase+1}: obstacles={obstacles}, guards={guards}")
            model.set_env(vec_env)

        model.learn(
            total_timesteps=phase_timesteps,
            reset_num_timesteps=False,
            callback=callback
        )
        save_path = f"ppo_gridworld_{obstacles}obs_{guards}guards"
        model.save(save_path)
        print(f"Saved model: {save_path}")

        vec_env.close()

    print("Curriculum training complete.")
