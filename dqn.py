# dqn.py
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        device=None,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=20000
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_steps = 0
        self.target_update_freq = target_update_freq

        # epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

    def select_action(self, state_np, eval_mode=False):
        """
        state_np: numpy array (state_dim,)
        returns: int action
        """
        eps = self.epsilon_final if eval_mode else self._epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.q_net(state_t)
        return int(qvals.argmax().item())

    def _epsilon(self):
        t = self.learn_steps
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
              max(0, (1 - t / self.epsilon_decay))
        return eps
    
    @property
    def epsilon(self):
        return self._epsilon()

    def store_transition(self, state, action, reward, next_state, done):
        # store numpy arrays / scalars
        self.replay.push(state.astype(np.float32), int(action), float(reward),
                         next_state.astype(np.float32) if next_state is not None else None,
                         bool(done))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None  # not enough samples

        transitions = self.replay.sample(self.batch_size)
        states = torch.FloatTensor(np.vstack(transitions.state)).to(self.device)
        actions = torch.LongTensor(transitions.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(transitions.reward).unsqueeze(1).to(self.device)
        non_final_mask = torch.BoolTensor([s is not None for s in transitions.next_state]).to(self.device)
        non_final_next_states = torch.FloatTensor(
            np.vstack([s for s in transitions.next_state if s is not None])
        ).to(self.device) if any(non_final_mask) else None
        dones = torch.FloatTensor(transitions.done).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # target: r + gamma * max_a' Q_target(s', a') for non-terminal
        next_q = torch.zeros(self.batch_size, 1, device=self.device)
        if non_final_next_states is not None:
            with torch.no_grad():
                next_q_vals = self.target_net(non_final_next_states)
                next_q[non_final_mask] = next_q_vals.max(1)[0].unsqueeze(1)

        q_target = rewards + (1.0 - dones) * (self.gamma * next_q)

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping optionally
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            'q_state_dict': self.q_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': getattr(self, 'epsilon', 0.1)  # save epsilon if exists
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path):
        import os
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
        d = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(d['q_state_dict'])
        self.target_net.load_state_dict(d['target_state_dict'])
        self.optimizer.load_state_dict(d['optimizer_state'])
        if 'epsilon' in d:
            self.epsilon = d['epsilon']
        print(f"Agent loaded from {path}")
