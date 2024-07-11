import os
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
import wandb

# Import your custom environment
from circle_navigation_v0 import *

wandb.login(key='ac7098cb1e72af1ef1563461abdef00127f089bb')

# Initialize wandb
wandb.init(
    project="circle_navigation"
)

# Initialize Ray
ray.init()

# Register the environment
medium_env = env(render_mode=None, difficulty='medium')

env_name = "circle_navigation_v0"
register_env(env_name, lambda config: PettingZooEnv(medium_env))

# Create a test environment to get observation and action spaces
test_env = PettingZooEnv(medium_env)
obs_space = test_env.observation_space
act_space = test_env.action_space

# Configure DQN
config = (
    DQNConfig()
    .environment(env=env_name)
    .rollouts(num_rollout_workers=4, rollout_fragment_length=32)
    .training(
        train_batch_size=256,
        lr=1e-4,
        gamma=0.99,
        n_step=1,
        target_network_update_freq=500,
        hiddens=[256, 256],
    )
    .multi_agent(
        policies={agent: (None, obs_space, act_space, {}) for agent in test_env.get_agent_ids()},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .framework("torch")
    .exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 100000,
        }
    )
)

# Define a custom callback for additional logging
class CustomCallback(ray.rllib.algorithms.callbacks.DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # Log additional metrics
        episode.custom_metrics["avg_distance_to_target"] = episode.last_info_for("agent_0").get("avg_distance_to_target", 0)
        episode.custom_metrics["num_conservative_actions"] = episode.last_info_for("agent_0").get("num_conservative_actions", 0)

# Add the custom callback to the config
config = config.callbacks(CustomCallback)

# Run the training
stop = {"timesteps_total": 1000000}
results = tune.run(
    "DQN",
    name="DQN_CircleNavigation",
    stop=stop,
    config=config.to_dict(),
    checkpoint_freq=10,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max",
    callbacks=[WandbLoggerCallback(
        project="circle_navigation",
        log_config=True,
    )],
)

# Get the best trial
best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
print(f"Best trial final reward: {best_trial.last_result['episode_reward_mean']}")

# Save the best checkpoint
best_checkpoint = best_trial.checkpoint.value
print(f"Best checkpoint: {best_checkpoint}")

# Finish the wandb run
wandb.finish()

ray.shutdown()