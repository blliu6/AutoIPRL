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
    env_name = 'C9'
    example = get_examples_by_name(env_name)
    load = True
    begin = timeit.default_timer()
    opts = {
        'example': example,
        'epsilon_step': 0.05,
        'num_episodes': 60,
        'epsilon': 0.5,
        'multiple_rewards': 1,
        'buffer_size': 20000,
        'batch_size': 1000,
        'lr': 1e-4
    }
    config = ProofConfig(**opts)

    multiple = 5
    env = Env(example, multiple=multiple)

    agent = DQN(config, load=load, double_dqn=True)

    if not load:
        num = 300
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
