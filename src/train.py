from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
import random

from copy import deepcopy

# Set up the environment with a maximum of 200 steps per episode
training_env = TimeLimit(
    env=HIVPatient(domain_randomization=False), 
    max_episode_steps=200
)

# Default hyperparameters
default_args = {
    "hidden_dim": 500,
    "n_hidden_layers": 5,
    "lr": 1e-3,
    "batch_size": 128,
    "nb_gradient_steps": 5,
    "update_target_freq": 400,
    "capacity": 10000,
    "gamma": 0.99,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_stop": 20000,
    "epsilon_delay": 100,
}

class ProjectAgent:
    def __init__(self, config=default_args):
        # Environment setup
        self.env = training_env
        self.n_episodes = 500
        self.n_episodes_steps = self.env._max_episode_steps

        # Dimensions and architecture
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.hidden_dim = config["hidden_dim"]
        self.n_hidden_layers = config["n_hidden_layers"]

        # Device selection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Main and target networks
        self.dqn_model = self.get_model(
            state_dim=self.state_dim, 
            n_actions=self.n_actions, 
            n_hidden_layers=self.n_hidden_layers, 
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.target_dqn = deepcopy(self.dqn_model).to(self.device)
        self.target_dqn.eval()

        # Optimization and loss
        self.lr = config["lr"]
        self.optim = optim.Adam(self.dqn_model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay and batch settings
        self.batch_size = config["batch_size"]
        self.replay_data = ReplayBuffer(config["capacity"], self.device)
        self.nb_gradient_steps = config["nb_gradient_steps"]
        self.update_target_freq = config["update_target_freq"]

        # Epsilon-greedy settings
        self.gamma = config["gamma"]
        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_stop = config["epsilon_stop"]
        self.epsilon_delay = config["epsilon_delay"]
        self.epsilon_decay_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        observation_t = torch.tensor(observation, dtype=torch.float32, device=self.device)
        q_vals = self.dqn_model(observation_t)
        return torch.argmax(q_vals).item()

    def save(self, path):
        torch.save(self.dqn_model.state_dict(), path)

    def load(self):
        model_path = "models/best_model.pth"
        self.dqn_model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

    def get_model(self, state_dim, n_actions, n_hidden_layers, hidden_dim):
        layers = nn.Sequential()
        layers.add_module("input", nn.Linear(state_dim, hidden_dim))
        layers.add_module("relu_in", nn.ReLU())
        for i in range(n_hidden_layers):
            layers.add_module(f"hidden_{i}", nn.Linear(hidden_dim, hidden_dim))
            layers.add_module(f"relu_{i}", nn.ReLU())
        layers.add_module("output", nn.Linear(hidden_dim, n_actions))
        return layers

    def gradient_step(self):
        if len(self.replay_data) > self.batch_size:
            X, A, R, Y, D = self.replay_data.sample(self.batch_size)

            # Compute the greedy actions for the next states using the current network
            best_next_actions = self.dqn_model(Y).argmax(dim=1).unsqueeze(1)

            # Evaluate Q-values using the target network
            with torch.no_grad():
                q_next = self.target_dqn(Y).gather(1, best_next_actions).squeeze(1)

            # Target Q-values
            target_q = R + self.gamma * q_next * (1 - D)

            # Current Q-values
            current_q = self.dqn_model(X).gather(1, A.long().unsqueeze(1))

            loss = self.loss_fn(current_q, target_q.unsqueeze(1))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def train(self):
        total_episode_rewards = []
        episode_count = 0
        cumulative_ep_reward = 0

        state, _ = self.env.reset()
        epsilon = self.epsilon_max
        global_step = 0
        record_reward = -np.inf

        while episode_count < self.n_episodes:
            if global_step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_decay_step)

            # Epsilon-greedy action selection
            use_random = np.random.rand() < epsilon
            action = self.act(state, use_random)

            # Interact with the environment
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.replay_data.push(state, action, reward, next_state, done)
            cumulative_ep_reward += reward

            # Perform several gradient steps
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target network periodically
            if global_step % self.update_target_freq == 0:
                self.target_dqn.load_state_dict(self.dqn_model.state_dict())

            global_step += 1

            if done or truncated:
                total_episode_rewards.append(cumulative_ep_reward)
                episode_count += 1
                print(
                    f"Episode {episode_count} - "
                    f"Reward: {cumulative_ep_reward:.3f}, epsilon: {epsilon:.3f}"
                )
                state, _ = self.env.reset()
                cumulative_ep_reward = 0

                # Save the model if it achieves a new best reward
                if total_episode_rewards[-1] > record_reward:
                    record_reward = total_episode_rewards[-1]
                    print(f"New best reward: {record_reward:.3f}")
                    self.save("models/model.pth")
            else:
                state = next_state

        return total_episode_rewards


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Convert batch elements to tensors
        return list(
            map(lambda x: torch.tensor(np.array(x), device=self.device),
                zip(*batch))
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    agent = ProjectAgent()
    print(agent.device)
    final_rewards = agent.train()
