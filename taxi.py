import gym
from collections import deque
from collections import defaultdict
import numpy as np
from agent import Agent
import time

env = gym.make("Taxi-v3")

# No. of possible actions
action_size = env.action_space.n
print(f"Action Space {env.action_space.n}")

# No. of possible states
space_size = env.observation_space.n
print(f"State Space {env.observation_space.n}")


def testing_without_learning():
    state = env.reset()
    total_rewards = 0

    def decode(i):
        out = list()
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        return reversed(out)

    while True:
        env.render()
        print(list(decode(state)))
        print("0: Down, 1: Up, 2: Right, 3: Left, 4: Pickup, 5: Dropoff")
        action = input("Select action: ")
        while action not in ["0", "1", "2", "3", "4", "5"]:
            action = input("Select action: ")
        action = int(action)
        next_state, reward, done, _ = env.step(action)
        print(f"Reward: {reward}")
        total_rewards = total_rewards + reward
        if done:
            print(f"Total reward: {total_rewards}")
            break
        state = next_state


def model_free_RL(Q, mode):
    agent = Agent(Q, mode)
    num_episodes = 100000
    sample_rewards = deque(maxlen=100)

    start_time = time.time()
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        eps = 1.0 / ((i_episode // 100) + 1)
        samp_reward = 0

        while True:
            action = agent.select_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            samp_reward += reward
            if done:
                sample_rewards.append(samp_reward)
                break
            state = next_state

        if i_episode >= 100:
            avg_reward = sum(sample_rewards) / len(sample_rewards)
            print(f"\rEpisode: {i_episode:6d}/{num_episodes} || Average reward: {avg_reward:7.2f} || eps: {eps:.5f}", end='')
    print(f"\n{'MC-control' if mode == 'mc_control' else 'Q-learning'} time: {time.time() - start_time:.2f} secs")


def testing_after_learning(Q):
    agent = Agent(Q)
    total_test_episode = 100
    rewards = list()

    for episode in range(total_test_episode):
        state = env.reset()
        episode_reward = 0
        eps = 0.00000001

        while True:
            action = agent.select_action(state, eps)
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                break
            state = new_state

    print(f"Avg reward: {sum(rewards) / total_test_episode}")


Q = defaultdict(lambda: np.zeros(action_size))

while True:
    print()
    print("1. Testing without learning")
    print("2. MC-control")
    print("3. Q-learning")
    print("4. Testing after learning")
    print("5. Exit")
    menu = input("Select: ")
    while menu not in ["1", "2", "3", "4", "5"]:
        menu = input("Select: ")
    menu = int(menu)
    if menu == 1:
        testing_without_learning()
    elif menu == 2:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "mc_control")
    elif menu == 3:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "q_learning")
    elif menu == 4:
        testing_after_learning(Q)
    elif menu == 5:
        break
