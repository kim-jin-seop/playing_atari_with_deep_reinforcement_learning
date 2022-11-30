import gym
from gym.spaces import Box
from gym.wrappers import Monitor

import cv2
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F

import networks
from memory import ReplayBuffer


# Image : Size, RGB to Gray
class FramePreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super(FramePreprocessing, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        return observation[:, :, None]


# Image : H W C -> C H W
class ChangeImageAxis(gym.ObservationWrapper):
    def __init__(self, env):
        super(ChangeImageAxis, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=(obs_shape[::-1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, -1, 0)

# for 4 Frame
class FromBuffer(gym.ObservationWrapper):
    def __init__(self, env):
        super(FromBuffer, self).__init__(env)
        self.temp = None
        self.observation_space = Box(env.observation_space.low.repeat(4, axis=0),
                                     env.observation_space.high.repeat(4, axis=0), dtype=np.float32)

    def reset(self):
        self.temp = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.temp[:-1] = self.temp[1:]
        self.temp[-1] = observation
        return self.temp


# image Scale [0 255] -> [0, 1]
class ImageScaling(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name: str, vidio_path):
    env = gym.make(env_name)
    env = FramePreprocessing(env)
    env = ChangeImageAxis(env)
    env = FromBuffer(env)
    env = ImageScaling(env)
    env = Monitor(env, vidio_path, force=True)
    return env


def experiment(env_name: str, action_num: int, learning_rate=1e-5,
               epochs: int = 10000, batch_size: int = 32, gamma: float = 0.98,
               eps_init: float = 1, eps_grad: float = 0.2, eps_min: float = 0.01,
               csv_name: str = 'test.csv', vidio_path: str = './monitor'):
    div = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = make_env(env_name, vidio_path=vidio_path)
    memory = ReplayBuffer(20000)

    q = networks.DQN(action_num).to(div)
    q_target = networks.DQN(action_num).to(div)

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    episode_list = []
    reward_list = []
    eps_list = []
    for epoch in range(epochs):
        print_score = 0.0
        eps = max(eps_min, eps_init - 0.01 * (epoch / (1 / eps_grad)))
        state = env.reset()

        while True:
            action = q.sample_action(torch.from_numpy(np.float32(state)).unsqueeze(0).to(div), eps)
            next_state, reward, finish, _ = env.step(action)
            memory.put(state, action, reward, next_state, finish)
            state = next_state
            print_score += reward

            if len(memory) > 10000:
                update(q, q_target, memory, optimizer, epoch, div, batch_size, gamma)

            if finish:
                break
        print("n_episode :{}, score : {:.1f}, eps : {:.1f}%".format(epoch, print_score, eps * 100))
        episode_list.append(epoch)
        reward_list.append(print_score)
        eps_list.append(eps * 100)

    df = pd.DataFrame({'episode': episode_list, 'reward': reward_list, 'epsilon': eps_list})
    df.to_csv(csv_name, index=False, mode='w')


def update(q, q_target, memory, optimizer, epoch, div, batch_size=32, gamma=0.98):
    if epoch % 1000:
        q_target.load_state_dict(q.state_dict())

    state, action, reward, next_state, finish = memory.sample(batch_size)
    state = torch.from_numpy(np.float32(state)).to(div)
    action = torch.from_numpy(np.int64(action)).to(div)
    reward = torch.from_numpy(reward).to(div)
    next_state = torch.from_numpy(next_state).to(div)
    finish = torch.from_numpy(finish).to(div)
    q_out = q(state)
    q_a = q_out.gather(1, action.unsqueeze(-1)).squeeze(-1)
    max_next_q = q_target(next_state).max(1)[0]

    target = reward + gamma * max_next_q * (1 - finish)
    loss = F.mse_loss(q_a, target.data.to(div))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
