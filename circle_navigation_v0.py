import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
from gymnasium import spaces
import functools
from collections import defaultdict
import math

class CircleNavigationEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "circle_navigation_v0",
    }

    def __init__(self, num_circles=10, num_agents=5, max_agents_per_circle=2, max_steps=100, render_mode=None, difficulty='medium'):
        self.num_circles = num_circles
        self._num_agents = num_agents  # Changed to a regular attribute
        self.max_agents_per_circle = max_agents_per_circle
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # Set up the figure for rendering
        if self.render_mode in ["human", "rgb_array"]:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_xlim(-1, self.num_circles)
            self.ax.set_ylim(-1, 1)
            self.ax.set_aspect('equal')
            self.ax.axis('off')

    @property
    def num_agents(self):
        return self._num_agents

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.steps = 0

        # Initialize agent positions randomly
        self.agent_positions = np.random.randint(0, self.num_circles, size=self._num_agents)

        # Assign target positions
        self.target_positions = np.random.randint(0, self.num_circles, size=self._num_agents)

        # Initialize agent positions based on difficulty
        if self.difficulty == 'easy':
            max_distance = 1
        elif self.difficulty == 'medium':
            max_distance = self.num_circles // 4
        else:  # hard
            max_distance = self.num_circles // 2

        self.agent_positions = np.clip(
            self.target_positions + np.random.randint(-max_distance, max_distance + 1, size=self._num_agents),
            0, self.num_circles - 1
        )

        # Initialize conservative strategy flags
        self.conservative = np.zeros(self._num_agents, dtype=bool)

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()

        return observations, infos

    def step(self, actions):
        self.steps += 1

        rewards = {agent: 0 for agent in self.agents}

        # Process actions
        for agent, action in actions.items():
            agent_idx = self.agent_name_mapping[agent]
            old_position = self.agent_positions[agent_idx]
            if action == 0:  # Move left
                self.agent_positions[agent_idx] = max(0, self.agent_positions[agent_idx] - 1)
            elif action == 1:  # Move right
                self.agent_positions[agent_idx] = min(self.num_circles - 1, self.agent_positions[agent_idx] + 1)
            elif action == 2:  # Stay
                self.conservative[agent_idx] = True
            else:
                raise ValueError(f"Invalid action {action} for agent {agent}")

            # Calculate intermediate reward
            new_position = self.agent_positions[agent_idx]
            target = self.target_positions[agent_idx]
            old_distance = abs(old_position - target)
            new_distance = abs(new_position - target)
            intermediate_reward = 0.1 * (old_distance - new_distance)  # Small reward for moving towards target
            rewards[agent] = intermediate_reward

        # Prepare observations and infos
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        # Check for truncation (max steps reached)
        truncations = {agent: self.steps >= self.max_steps for agent in self.agents}

        # Check for termination (all agents conservative)
        terminations = {agent: np.all(self.conservative) for agent in self.agents}

        # Calculate final rewards if game is over
        if any(truncations.values()) or any(terminations.values()):
            final_rewards = self._calculate_final_rewards()
            for agent in self.agents:
                rewards[agent] += final_rewards[agent]

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()

        return observations, rewards, terminations, truncations, infos

    def _calculate_final_rewards(self):
        final_rewards = {}
        for i, agent in enumerate(self.agents):
            # Base reward for reaching target
            if self.agent_positions[i] == self.target_positions[i]:
                final_rewards[agent] = 10
            else:
                final_rewards[agent] = -abs(self.agent_positions[i] - self.target_positions[i])

            # Penalty for overcrowded circles
            unique, counts = np.unique(self.agent_positions, return_counts=True)
            overcrowded = unique[counts > self.max_agents_per_circle]
            if self.agent_positions[i] in overcrowded:
                final_rewards[agent] -= 20  # Major penalty for being in an overcrowded circle

        return final_rewards

    def _get_obs(self, agent_idx):
        return {
            "position": self.agent_positions[agent_idx],
            "target": self.target_positions[agent_idx],
            "all_positions": self.agent_positions.tolist(),
        }

    def render(self):
        if self.render_mode == "human":
            plt.pause(0.1)
            return None
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.ax.clear()
        self.ax.set_xlim(-1, self.num_circles)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Group agent positions
        agent_groups = defaultdict(list)
        for i, pos in enumerate(self.agent_positions):
            agent_groups[pos].append(i)

        # Draw circles and agents
        for i in range(self.num_circles):
            # Draw main circle
            main_circle = Circle((i, 0), 0.4, fill=False)
            self.ax.add_patch(main_circle)

            # Draw number of agents in the circle
            num_agents = len(agent_groups[i])
            self.ax.text(i, 0, str(num_agents), ha='center', va='center', fontweight='bold')

            # Draw agents around the circle
            if num_agents > 0:
                radius = 0.3  # Radius for agent placement
                for j, agent_idx in enumerate(agent_groups[i]):
                    angle = 2 * math.pi * j / num_agents - math.pi / 2  # Start from the top and go clockwise
                    x = i + radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    agent = Circle((x, y), 0.1, fill=True, color='blue', alpha=0.7)
                    self.ax.add_patch(agent)
                    self.ax.text(x, y, f'A{agent_idx}', ha='center', va='center', fontsize=8)

        # Group target positions
        target_groups = defaultdict(list)
        for i, pos in enumerate(self.target_positions):
            target_groups[pos].append(i)

        # Draw target positions and their labels
        for pos, indices in target_groups.items():
            # Place target labels in a column below the circle
            for i, idx in enumerate(indices):
                y_offset = -0.5 - 0.2 * i  # Adjust vertical spacing here
                self.ax.text(pos, y_offset, f'T{idx}', ha='center', va='center', color='red')

        self.fig.canvas.draw()
        if self.render_mode == "rgb_array":
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({
            "position": spaces.Discrete(self.num_circles),
            "target": spaces.Discrete(self.num_circles),
            "all_positions": spaces.Box(low=0, high=self.num_circles - 1, shape=(self._num_agents,), dtype=int),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)  # 0: Move left, 1: Move right, 2: Stay


def env(render_mode=None, difficulty='medium'):
    env = CircleNavigationEnv(render_mode=render_mode, difficulty=difficulty)
    env = parallel_to_aec(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
