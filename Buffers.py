class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (torch.tensor(state, dtype=torch.float32),
                                 torch.tensor(action, dtype=torch.long),
                                 torch.tensor(reward, dtype=torch.float32),
                                 torch.tensor(next_state, dtype=torch.float32),
                                 torch.tensor(done, dtype=torch.float32))
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.stack(actions), torch.stack(rewards),
                torch.stack(next_states), torch.stack(dones))

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def push(self, state, action):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((torch.tensor(state, dtype=torch.float32),
                                torch.tensor(action, dtype=torch.long)))
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (torch.tensor(state, dtype=torch.float32),
                                    torch.tensor(action, dtype=torch.long))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions = zip(*batch)
        return torch.stack(states), torch.stack(actions)

    def __len__(self):
        return len(self.buffer)