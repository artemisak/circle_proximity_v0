from circle_navigation_v0 import *
import numpy as np

# For human rendering
medium_env = env(render_mode="human", difficulty='medium')
medium_env.reset()

for agent in medium_env.agent_iter():
    observation, reward, termination, truncation, info = medium_env.last()
    action = np.random.randint(3) if not termination and not truncation else None
    medium_env.step(action)
    medium_env.render()  # This will display the game state

medium_env.close()
