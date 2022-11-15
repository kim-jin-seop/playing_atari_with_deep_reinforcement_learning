from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, max_buffer):
        self.buffer = deque(maxlen=max_buffer)

    def put(self, s, a, r, n_s, done):
        self.buffer.append((s, a, r, n_s, done))

    def sample(self, n):
        batch_index = random.sample(self.buffer, n)
        batch = zip(*[self.buffer[i] for i in batch_index])
        s_list, a_list, r_list, n_s_list, done_list = batch

        return (torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list),
                torch.tensor(r_list), torch.tensor(n_s_list, dtype=torch.float),
                torch.tensor(done_list))

    def __len__(self):
        return len(self.buffer)
