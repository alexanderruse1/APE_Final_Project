import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Actor-Critic model that jointly learns a policy (actor) and a value function (critic)
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        # Actor: outputs a probability distribution over actions using a softmax policy
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64), # First hidden layer
            nn.ReLU(), # Non-linearity
            nn.Linear(64, n_actions), # Output layer
            nn.Softmax(dim=-1) # Normalize to a probability distribution
        )
        # Critic: estimates the state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64), # First hidden layer
            nn.ReLU(),
            nn.Linear(64, 1) # Scalar output for value estimate
        )

    def forward(self, x):
        value = self.critic(x) # Value function V(s)
        probs = self.actor(x)  # Policy π(a|s)
        return probs, value
# Training loop for Actor-Critic
def train(env, episodes=10):
    # Initialize model with observation size and number of actions
    model = ActorCritic(input_dim=env.n_containers, n_actions=env.n_containers + 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    gamma = 0.99 # Discount factor for future rewards

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            input_vec = obs['volume'] # Observed container volumes
            state = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0) # Add batch dimension
            probs, value = model(state)
            # Sample an action from the policy distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = action.item()
            # Take action in the environment
            next_obs, reward, done, _ = env.step(action_idx)
            next_state = torch.tensor(next_obs['volume'], dtype=torch.float32).unsqueeze(0)
            # Compute target for value function (one-step TD target)
            _, next_value = model(next_state)
            advantage = reward + gamma * next_value.item() - value.item()
            # Compute actor and critic losses
            actor_loss = -dist.log_prob(action) * advantage # Policy gradient term
            critic_loss = advantage ** 2 # Mean squared TD error

            loss = actor_loss + critic_loss
            # Backpropagation and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
             # Move to next state
            obs = next_obs
            ep_reward += reward
        print(f"Episode {ep + 1}: Total Reward = {ep_reward}")
