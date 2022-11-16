import gym
import torch

import networks
from memory import ReplayBuffer
from gym.spaces import Box
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np

memory_size = 100000
print_interval = 100

class FrmDwSmpl(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrmDwSmpl, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class Img2Trch(gym.ObservationWrapper):
    def __init__(self, env):
        super(Img2Trch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=(obs_shape[::-1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class FrmBfr(gym.ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrmBfr, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),
                                     obs_space.high.repeat(num_steps, axis=0), dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class NormFlts(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = FrmDwSmpl(env)
    env = Img2Trch(env)
    env = FrmBfr(env, 4)
    env = NormFlts(env)
    return env


def train(env_name: str, action_count: int, learning_rate=1e-5,
          epochs: int = 3000, batch_size: int = 32, gamma: float = 0.98):
    env = make_env(env_name)

    q = networks.DQN(action_count)
    q_target = networks.DQN(action_count)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(memory_size)

    print_score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        eps = max(0.01, 0.08 - 0.01 * (epoch / 200))
        state = env.reset()
        finish = False

        while not finish:
            action = q.sample_action(torch.from_numpy(state).float().unsqueeze(0), eps)
            env.render()
            next_state, reward, finish, _ = env.step(action)
            f_mask = 0.0 if finish else 1.0
            memory.put(state, action, reward, next_state, f_mask)
            state = next_state
            print_score += reward

        if len(memory) > 2000:
            update(q, q_target, memory, optimizer, batch_size, gamma)

        if epoch != 0 and epoch % print_interval == 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                epoch, print_score / print_interval, len(memory), eps * 100))
            print_score = 0


def update(q, q_target, memory, optimizer, batch_size=32, gamma=0.98):
    for i in range(10):
        state, action, reward, next_state, finish = memory.sample(batch_size)
        q_out = q(state)
        q_a = torch.gather(q_out, 1, action.unsqueeze(-1))

        max_q_prime = q_target(next_state).max(1)[0].unsqueeze(1)
        target = reward.unsqueeze(-1) + gamma * max_q_prime * finish.unsqueeze(-1)
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train("Pong-v4", 4)