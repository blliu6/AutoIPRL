import torch


class ProofConfig:
    example = None
    lr = 1e-5
    num_episodes = 100
    units = 128
    gamma = 0.98
    epsilon = 0.9
    epsilon_step = 0.01
    target_update = 20
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 1000
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    multiple_rewards = 10

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
