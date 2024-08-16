import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return list(state), list(action), list(reward), list(next_state), list(done)

    def size(self):
        return len(self.buffer)


def sample_data(env, batch_size, replay_buffer):
    batch = env.replay_buffer.size()
    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size - batch)
    b_s_, b_a_, b_r_, b_ns_, b_d_ = env.replay_buffer.sample(batch)
    b_s.extend(b_s_)
    b_a.extend(b_a_)
    b_r.extend(b_r_)
    b_ns.extend(b_ns_)
    b_d.extend(b_d_)
    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
    return transition_dict


if __name__ == '__main__':
    replay = ReplayBuffer(10)
    for i in range(10):
        replay.add([i, i + 1], [i, i + 2], i, [i + 3, i + 4], [i])

    a, b, c, d, e = replay.sample(30)
    print(type(a), type(b), type(c), type(d), type(e))
    print(a, b, c, d, e)
