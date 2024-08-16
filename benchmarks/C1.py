import random
import timeit

import numpy as np
import torch

from proof.Env import Env
from proof.Example import get_examples_by_name
from proof.dqn import DQN, train_off_policy_agent
from proof.proof_config import ProofConfig
from proof.reappear import reappear


def main():
    env_name = 'C1'
    example = get_examples_by_name(env_name)
    load = False
    begin = timeit.default_timer()
    opts = {
        'example': example,
        'epsilon_step': 0.1,
        'num_episodes': 30,
        'epsilon': 0.7,
        'buffer_size': 10000,
        'batch_size': 500,
        'unit': 64,
        'lr': 1e-4,
        'device': torch.device('cuda:0')
    }
    config = ProofConfig(**opts)

    env = Env(example)

    agent = DQN(config, load=load, double_dqn=True)

    if not load:
        num = 100
        train_off_policy_agent(env, agent, config, num=num)
        end = timeit.default_timer()
        print(f'Total time: {end - begin}s')
    else:
        reappear(agent, env)


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    main()
