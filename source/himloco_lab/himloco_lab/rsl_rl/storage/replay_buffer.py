import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states):
        """Add new states to memory."""

        # 获取输入状态的行数
        num_states = states.shape[0]
        # 计算开始和结束的索引
        start_idx = self.step
        end_idx = self.step + num_states

        # 如果结束索引超过缓冲区大小
        if end_idx > self.buffer_size:
            # 将新状态放入缓冲区的前半部分
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            # 将剩余的新状态放入缓冲区的后半部分
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            # 直接将新状态放入缓冲区
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        # 更新样本数量
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))

        # 更新步长索引
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        # 遍历每一个mini-batch
        for _ in range(num_mini_batch):
            # 从所有样本中随机选择mini_batch_size个样本的索引
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            # 生成一个包含当前状态和下一个状态的元组，并将其移动到指定的设备上
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))
