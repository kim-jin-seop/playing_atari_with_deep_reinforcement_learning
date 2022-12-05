
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def put(self, s, a, r, n_s, done):
        self.buffer.append((s, a, r, n_s, done))

    def sample(self, n):
        batch_index = np.random.choice(len(self.buffer), n, False)
        batch = zip(*[self.buffer[i] for i in batch_index])
        s_list, a_list, r_list, n_s_list, done_list = batch

        return (np.array(s_list), np.array(a_list), np.array(r_list, dtype=np.float32),
                np.array(n_s_list), np.array(done_list, dtype=np.uint8))

    def clean(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
