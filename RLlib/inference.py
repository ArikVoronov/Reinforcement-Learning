from time import sleep
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig

# path_to_checkpoint=r'F:\Study\Programming\Machine Learning\Projects\rl\ray_results\2023-05-20_13-53-48__Taxi-v3\checkpoint_000015'
path_to_checkpoint = r'F:\Study\Programming\Machine Learning\Projects\rl\ray_results\2023-05-20_19-11-09__CartPole-v1\checkpoint_000015'
env_name = "CartPole-v1"  # Taxi-v3 / CartPole-v1
env = gym.make(env_name, render_mode='rgb_array')
algo = PPOConfig().training(model={"fcnet_hiddens": [64, 64]}).environment(env_name).build()
algo.restore(path_to_checkpoint)
episode_reward = 0
terminated = truncated = False

if gymnasium:
    obs, info = env.reset()
else:
    obs = env.reset()

fig, ax = plt.subplots()


def update(frame):
    ax.clear()
    ax.imshow(frame)

render_real_time=True
img = None
render_images = []
while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    if gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, info = env.step(action)

    episode_reward += reward
    print(episode_reward)

    render_img = env.render()
    render_images.append(render_images)
    if render_real_time:
        if img is None:
            img = plt.imshow(render_img)
        else:
            img.set_data(render_img)
        plt.pause(.001)
        plt.draw()
print(f'Done {terminated}')

if not render_real_time:
    print('Rendering')
    for render_img in render_images:
        print(np.array(render_img))
        cv2.imshow("Image", render_img)
        cv2.waitKey(1)  # Wait for a key press (1 millisecond)

        # Close the OpenCV window at the end
    cv2.destroyAllWindows()


env.close()
