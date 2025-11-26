import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from PokerEnv import Poker5EnvFull
from Buffers import ReplayBuffer, ReservoirBuffer  # tus buffers

# -----------------------
# Redes Neuronales
# -----------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


# -----------------------
# Agente NFSP
# -----------------------
class PokerAgentNFSP:
    def __init__(self, state_dim, action_dim, epsilon=0.1, reservoir_capacity=100000, replay_capacity=100000):
        self.action_dim = action_dim
        self.epsilon = epsilon  # para exploración ε-greedy

        # Redes
        self.q_network = QNetwork(state_dim, action_dim)
        self.policy_network = PolicyNetwork(state_dim, action_dim)

        # Buffers
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.reservoir_buffer = ReservoirBuffer(capacity=reservoir_capacity)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-3)

    # -----------------------
    # Selección de acción
    # -----------------------
    def act(self, state, legal_actions=None, use_policy=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if use_policy:
            # Elegir acción según policy network (SL)
            probs = self.policy_network(state_tensor).detach().numpy()[0]
            if legal_actions is not None:
                mask = np.zeros_like(probs)
                mask[legal_actions] = 1
                probs = probs * mask
                probs = probs / probs.sum()
            return np.random.choice(len(probs), p=probs)
        else:
            # ε-greedy sobre Q-network
            if random.random() < self.epsilon:
                return random.choice(legal_actions) if legal_actions else random.randint(0, self.action_dim-1)
            else:
                q_values = self.q_network(state_tensor).detach().numpy()[0]
                if legal_actions is not None:
                    q_values = [q if i in legal_actions else -np.inf for i,q in enumerate(q_values)]
                return int(np.argmax(q_values))

    # -----------------------
    # Guardar experiencias
    # -----------------------
    def store_rl(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def store_sl(self, state, action):
        self.reservoir_buffer.push(state, action)

    # -----------------------
    # Entrenamiento
    # -----------------------
    def train_q_network(self, batch_size=64, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.q_network(next_states).max(1)[0]
        target = rewards + gamma * next_q * (1 - dones)
        loss = F.mse_loss(q_values, target.detach())
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def train_policy_network(self, batch_size=64):
        if len(self.reservoir_buffer) < batch_size:
            return
        states, actions = self.reservoir_buffer.sample(batch_size)
        logits = self.policy_network(states)
        loss = F.cross_entropy(logits, actions)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

def evaluate_agent(agent, env, num_eval_episodes=10):
    total_reward = 0
    for _ in range(num_eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            legal_actions = env.get_legal_actions(env.agent_id)
            state_vector = np.concatenate([
                obs["hero_hand"],
                obs["board"],
                obs["stacks"],
                obs["pot"],
                obs["current_bet"]
            ]).astype(np.float32)

            action = agent.act(state_vector, legal_actions=legal_actions, use_policy=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
    return total_reward / num_eval_episodes

if __name__ == "__main__"::
    env = Poker5EnvFull()
    agent = PokerAgentNFSP(state_dim=19, action_dim=len(env.ACTIONS))
    
    num_episodes = 1000
    eval_every = 50

    for episode in range(1, num_episodes + 1):
        obs, _ = env.partial_reset()
        done = False

        while not done:
            legal_actions = env.get_legal_actions(env.agent_id)
            state_vector = np.concatenate([
                obs["hero_hand"], 
                obs["board"], 
                obs["stacks"], 
                obs["pot"], 
                obs["current_bet"],
                obs["active_players"]
            ]).astype(np.float32)  # Convertimos la obs a vector

            # Elegir acción
            action = agent.act(state_vector, legal_actions=legal_actions, use_policy=False)

            # Ejecutar acción
            next_obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward

            next_state_vector = np.concatenate([
                next_obs["hero_hand"], 
                next_obs["board"], 
                next_obs["stacks"], 
                next_obs["pot"], 
                next_obs["current_bet"]
            ]).astype(np.float32)

            # Guardar experiencias
            agent.store_rl(state_vector, action, reward, next_state_vector, done)
            agent.store_sl(state_vector, action)

            # Entrenar redes
            agent.train_q_network()
            agent.train_policy_network()

            obs = next_obs

        reward_history.append(episode_reward)

        if episode % eval_every == 0:
            
            avg_reward = evaluate_agent(agent, env, num_eval_episodes=50)
            print(f"Episode {episode}: Avg eval reward={avg_reward:.2f}")
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.q_network.state_dict(), "best_q_network.pth")
                torch.save(agent.policy_network.state_dict(), "best_policy_network.pth")
                print(f"  Nuevo mejor modelo guardado con avg reward {best_avg_reward:.2f}")