import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proof.ReplayBuffer import ReplayBuffer, sample_data
from proof.proof_config import ProofConfig


def initial_preparation(env):
    while True:
        state, done = env.reset()
        if done:
            break
        truncated = False
        while not done and not truncated:
            action = np.random.randint(len(env.action))
            next_state, reward, done, truncated, info = env.step(action)
    env.update_parameters()


class Qnet(nn.Module):
    def __init__(self, s_dim, dense=4, units=128):
        super().__init__()
        self.seq = nn.Sequential()
        s = s_dim
        for i in range(dense):
            self.seq.add_module(f'dense{i}', nn.Linear(s, units))
            self.seq.add_module(f'relu{i}', nn.ReLU())
            s = units
        self.seq.add_module(f'dense_{dense}', nn.Linear(units, 1))

    def forward(self, x):
        return self.seq(x)


class DQN:
    def __init__(self, config: ProofConfig, state_dim=2, action_dim=1, load=False, double_dqn=False):
        self.action_dim = action_dim
        self.device = config.device
        self.q_net = Qnet(state_dim + action_dim, dense=4, units=config.units).to(self.device)
        self.target_q_net = Qnet(state_dim + action_dim, dense=4, units=config.units).to(self.device)
        # self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.lr)
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.target_update = config.target_update
        self.count = 0
        self.steps = 1e6
        self.name = config.example.name
        self.double_dqn = double_dqn

        if load:
            self.q_net.load_state_dict(torch.load(f'../model/{self.name}.pth', map_location=self.device))
            print('Parameters loaded successfully!')
        else:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def take_action(self, state, action_len):
        action = [[i] for i in range(action_len)]
        state = state[1]
        if np.random.random() > self.epsilon:
            pos = np.random.randint(action_len)
        else:
            state = np.concatenate((np.array([state] * action_len), action), axis=1)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            pos = self.q_net(state).argmax().item()
        return pos

    def get_next_q(self, next_states_original, next_state, action_map):
        if self.double_dqn:
            act = []
            for i, state in enumerate(next_state):
                action = [[i] for i in range(action_map[next_states_original[i]])]
                state = np.concatenate((np.array([state] * len(action)), action), axis=1)
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                act.append(self.q_net(state).argmax().reshape(-1, 1))
            act = torch.cat(act, 0)
            input_state = torch.cat([torch.tensor(np.array(next_state), dtype=torch.float).to(self.device), act], 1)
            res = self.target_q_net(input_state)
            return res
        else:
            res = []
            for i, state in enumerate(next_state):
                action = [[i] for i in range(action_map[next_states_original[i]])]
                state = np.concatenate((np.array([state] * len(action)), action), axis=1)
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                res.append(self.target_q_net(state).max().reshape(-1, 1))
            return torch.cat(res, 0)

    def update(self, transition_dict, env):
        states_ = transition_dict['states']
        next_states_ = transition_dict['next_states']

        states_original, states = [item[0] for item in states_], [item[1] for item in states_]
        next_states_original, next_states = [item[0] for item in next_states_], [item[1] for item in next_states_]

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_input = torch.cat((states, actions), dim=1)
        q_values = self.q_net(q_input)
        max_next_q_values = self.get_next_q(next_states_original, next_states, env.map)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def save(self):
        torch.save(self.q_net.state_dict(), f'../model/{self.name}.pth')
        print('Model saved successfully!')


def train_off_policy_agent(env, agent, config: ProofConfig, multiple_rewards=10, num=500):
    initial_preparation(env)

    min_episode = 0
    replay_buffer = ReplayBuffer(config.buffer_size)
    env.max_episode = num
    for i_episode in range(config.num_episodes):
        # save model
        if agent.epsilon >= 1:
            state, done = env.reset()
            truncated = False
            while not done and not truncated:
                action = agent.take_action(state, len(env.action))
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                if done and agent.steps > info:
                    agent.steps, min_episode = info, i_episode
                    print(f'Proof steps:{agent.steps}, episode:{min_episode}')
                    agent.save()

        if i_episode % 5 == 0 and agent.epsilon < 1:
            agent.epsilon = min(agent.epsilon + config.epsilon_step, 1)

        episode_return = 0
        state, done = env.reset()
        print(f'Episode:{i_episode},len_memory:{env.len_memory},len_action:{len(env.action)}')

        truncated = False
        while not done and not truncated:
            action = agent.take_action(state, len(env.action))
            next_state, reward, done, truncated, info = env.step(action)

            if reward > 0:
                for i in range(multiple_rewards - 1):
                    replay_buffer.add(state, action, reward, next_state, done)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            if replay_buffer.size() > config.minimal_size:
                transition_dict = sample_data(env, config.batch_size, replay_buffer)
                agent.update(transition_dict, env)

        print(f'Sum of reward: {episode_return},agent_steps: {agent.steps}')
    print(f'Minimum number of proof steps:{agent.steps}, Minimum episode:{min_episode}')
