import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.ape_algorithm.forecast_ape import ForecastAPEAgent
# Forecast-Augmented Actor-Critic Training

class ForecastAugmentedActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value


def train_forecast_augmented(env_config, episodes=10, k=3, lambda_=0.2):
  # Extract container IDs and reward curve parameters
    container_ids = env_config.enabled_containers
    reward_params = {
        cid: {
            "peaks": env_config.containers[cid].reward.peaks,
            "heights": env_config.containers[cid].reward.heights,
            "widths": env_config.containers[cid].reward.widths
        }
        for cid in container_ids
    }
    # Map of maximum volumes for overflow risk calculation
    vmax_map = {cid: env_config.containers[cid].max_volume for cid in container_ids}
    # ForecastAPEAgent is used to compute forecasted priority scores
    forecast_agent = ForecastAPEAgent(container_ids, reward_params, vmax_map, lambda_=lambda_, k=k)
    # Initialize Actor-Critic model that uses both volume and forecasted priorities as input
    model = ForecastAugmentedActorCritic(input_dim=len(container_ids)*2, n_actions=len(container_ids)+1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    gamma = 0.99

    rewards = []
    for ep in range(episodes):
        env = ContainerEnv(env_config)
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
           # Combine raw volume and forecasted priorities into augmented input
            volumes = dict(zip(env.container_ids, obs["volume"]))
            priorities = forecast_agent.get_priorities(volumes)
            input_vec = np.array([[volumes[cid], priorities[cid]] for cid in container_ids]).flatten()
            state = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
           # Forward pass through the actor and critic networks
            probs, value = model(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = action.item()
           # Take action in the environment and observe next state and reward
            obs, reward, done, _ = env.step(action_idx)
           # Prepare next state input for value estimation
            volumes_next = dict(zip(env.container_ids, obs["volume"]))
            input_vec_next = np.array([[volumes_next[cid], priorities[cid]] for cid in container_ids]).flatten()
            next_state = torch.tensor(input_vec_next, dtype=torch.float32).unsqueeze(0)
           # Compute TD(0) advantage
            _, next_value = model(next_state)
            advantage = reward + gamma * next_value.item() - value.item()
           # Compute actor and critic losses
            actor_loss = -dist.log_prob(action) * advantage
            critic_loss = advantage ** 2
            loss = actor_loss + critic_loss
           # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")
    return rewards
