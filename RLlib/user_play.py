# import gymnasium as gym
import gym
import keyboard
import threading
import time
import matplotlib.pyplot as plt

# Create the Gym environment
env = gym.make('CartPole-v1')
# env = gym.make('Taxi-v3', render_mode='rgb_array')
env.reset()

action_dict = {
    'up': 1,
    'down': 0,
    'right': 2,
    'left': 3,
    'e': 4,
    'r': 5,
}

action_dict = {
    'right': 1,
    'left': 0,
}
def get_user_input():
    global user_action
    while True:
        user_action = 0
        for arrow_key in action_dict.keys():
            if keyboard.is_pressed(arrow_key):
                user_action = action_dict[arrow_key]
                print(arrow_key,user_action)

                break


# Create a thread for user input
input_thread = threading.Thread(target=get_user_input)
input_thread.daemon = True
input_thread.start()
time.sleep(0.1)

user_action = 0
total_reward = 0
img = None
try:
    # Run the environment
    while True:
        obs, reward, done, _ = env.step(user_action)
        total_reward += reward

        if done:
            print(f'done, reward: {total_reward}')
            env.reset()
            total_reward = 0
        time.sleep(0.1)  # Delay to control the speed of the simulation

        env.render()

except KeyboardInterrupt:
    pass

# Clean up
env.close()
