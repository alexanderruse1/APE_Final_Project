import gym
import numpy as np
from gym import spaces
from dataclasses import dataclass, field
from typing import List, Dict

# Why use dataclasses for list structures?
# Allows modular, declarative definitions of complex structured inputs (like reward shapes)
# Supports programmatic configuration generation, validation, and reuse
# Enables nested configurations for clean and scalable environments

# Configuration Classes
# Defines the shape of the reward function as a mixture of Gaussian-like peaks.
@dataclass
class RewardParam:
    peaks: List[float]
    heights: List[float]
    widths: List[float]

# Defines per-container stochastic behavior and reward profile
@dataclass
class ContainerConfig:
    mu: float # Mean fill rate (controls stochastic dynamics)
    sigma: float  # Std dev of fill rate (controls noise in fill)
    max_volume: float  # Maximum allowable volume (capacity threshold)
    bale_size: float # Amount removed when emptied
    press_offset: float #not used in the code
    press_slope: float #not used in the code
    reward: RewardParam # Reward curve parameters

# Defines global environment settings and container collection
@dataclass
class EnvConfig:
    max_episode_length: int = 1500 # Total time steps in an episode
    timestep: int = 120 # Simulated seconds per step
    enabled_containers: List[str] = field(default_factory=list) # Containers to include
    n_presses: int = 2 # Number of containers we can empty per step
    min_starting_volume: float = 0 # Lower bound for initial fill
    max_starting_volume: float = 30 # Upper bound for initial fill
    failure_penalty: float = -1  # Penalty for overflow
    min_reward: float = -0.1 # Small penalty for waiting
    containers: Dict[str, ContainerConfig] = field(default_factory=dict)

# Container Environment
class ContainerEnv(gym.Env):
    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        self.container_ids = config.enabled_containers
        self.n_containers = len(self.container_ids)
        self.n_presses = config.n_presses
        self.max_volume = max(cfg.max_volume for cfg in config.containers.values())
        self.episode_length = config.max_episode_length
        self.t = 0
# Randomize starting volume for each container
        self.volume = np.random.uniform(config.min_starting_volume,
                                        config.max_starting_volume,
                                        size=self.n_containers)
# Sample container-specific stochastic fill rates from a normal distribution
        self.fill_rates = np.array([
            np.random.normal(loc=config.containers[cid].mu,
                             scale=config.containers[cid].sigma)
            for cid in self.container_ids
        ])
# Discrete action space: one action per container + one "wait" action
        self.action_space = spaces.Discrete(self.n_containers + 1)
# Observation space: continuous vector of volumes (one per container)
        self.observation_space = spaces.Box(low=0, high=self.max_volume,
                                            shape=(self.n_containers,), dtype=np.float32)

    def reset(self):
# Resample starting volumes and stochastic fill rates
        self.volume = np.random.uniform(self.config.min_starting_volume,
                                        self.config.max_starting_volume,
                                        size=self.n_containers)
        self.fill_rates = np.array([
            np.random.normal(loc=self.config.containers[cid].mu,
                             scale=self.config.containers[cid].sigma)
            for cid in self.container_ids
        ])
        self.t = 0
        return {"volume": self.volume.copy()}

    def _reward_fn(self, cid: str, volume: float) -> float:
# Computes total reward from all reward peaks using a sum of Gaussians
        reward = 0
        params = self.config.containers[cid].reward
        for h, p, w in zip(params.heights, params.peaks, params.widths):
            reward += h * np.exp(-((volume - p) ** 2) / (2 * w ** 2))
        return reward

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_length
        reward = 0
# Update volumes using stochastic fill
        self.volume += self.fill_rates

        if action < self.n_containers:
# Empty selected container and collect reward based on current volume
            cid = self.container_ids[action]
            v = self.volume[action]
            reward = self._reward_fn(cid, v)
            self.volume[action] = 0
        else:
 # Apply mild penalty for taking no action
            reward = -0.01
 # If any container overflows, apply harsh penalty and end episode
        if np.any(self.volume > self.max_volume):
            reward = self.config.failure_penalty
            done = True

        return {"volume": self.volume.copy()}, reward, done, {}

    def render(self, mode='human'):
        print(f"Step {self.t}, Volumes: {self.volume}")

# Example Configuration
env_config = EnvConfig(
    enabled_containers=["C1-20", "C1-30", "C1-60", "C1-70", "C1-80"],
    containers={
        "C1-20": ContainerConfig(
            mu=0.005767754387396311, sigma=0.055559018416836935, max_volume=40,
            bale_size=27, press_offset=106.798502, press_slope=264.9,
            reward=RewardParam(peaks=[26.71], heights=[1], widths=[2])
        ),
        "C1-30": ContainerConfig(
            mu=0.003911622673679469, sigma=0.0298246737197056, max_volume=40,
            bale_size=12.5, press_offset=95.399, press_slope=149.88,
            reward=RewardParam(peaks=[26.52, 17.68, 8.84], heights=[1, 0.3, 0.1], widths=[2.5, 0.5, 0.25])
        ),
        "C1-60": ContainerConfig(
            mu=0.0019084442226913933, sigma=0.024947588621871703, max_volume=40,
            bale_size=8.5, press_offset=65.998499, press_slope=191.64,
            reward=RewardParam(peaks=[28.78, 21.61, 14.34, 7.17], heights=[1, 0.4, 0.2, 0.1], widths=[2.5, 0.5, 0.25, 0.125])
        ),
        "C1-70": ContainerConfig(
            mu=0.0035737010399693745, sigma=0.029566378732721433, max_volume=40,
            bale_size=10.5, press_offset=56.398501, press_slope=172.32,
            reward=RewardParam(peaks=[25.93, 17.28, 8.64], heights=[1, 0.25, 0.1], widths=[2.5, 0.5, 0.25])
        ),
        "C1-80": ContainerConfig(
            mu=0.008142898729319127, sigma=0.1227266060811535, max_volume=40,
            bale_size=12.5, press_offset=53.999001, press_slope=176.34,
            reward=RewardParam(peaks=[24.75, 12.37], heights=[1, 0.3], widths=[2, 0.5])
        ),
    }
)
