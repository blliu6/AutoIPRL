import multiprocessing as mp
from multiprocessing import Pool

import cvxpy as cp
import numpy as np
import sympy as sp

from proof.Example import Example
from proof.ReplayBuffer import ReplayBuffer
from proof.mapping import mul_polynomial_with_fft, get_map
from proof.monomials_generate import monomials


class Env:
    def __init__(self, example: Example, max_episode=500, multiple=1):
        self.n = example.n  # the number of variable
        self.deg = example.obj_deg  # the highest degree of polynomial
        self.l = example.l
        self.poly, self.poly_list = monomials(self.n, self.l)
        self.sp_poly = np.array([sp.sympify(e) for e in self.poly])
        self.len_vector = len(self.poly)
        self.objective = self.get_objective(example.objective)

        # fft_poly
        self.poly_map = [get_map(e, self.l + 1) for e in self.poly_list]
        self.dic_forward = dict(zip(range(self.len_vector), self.poly_map))
        self.dic_reverse = dict(zip(self.poly_map, range(self.len_vector)))
        self.max_map = max(self.poly_map)

        self.M, self.M_, self.A = None, None, None
        self.M_copy, self.A_copy = None, None
        self.M_deg_map = {}
        self.first_deg_pos = -1

        self.max_episode = max_episode
        self.episode = 0
        self.s = None
        self.memory, self.memory_action = None, None
        self.len_memory = 0
        self.coefficient_matrix = None
        self.set_memory, self.set_action, self.set_M = None, None, None
        self.last_gamma, self.gamma0, self.last_len = None, None, None
        self.state = None
        self.map, self.pos = {}, {}
        self.tuple_memory = []
        self.action = None
        self.origin_state = None
        self.reverse = {}
        print('Initialization starts!')
        self.memory_initialization()
        print('Initialization completed!')
        print(f'Length of vector: {self.len_vector}')
        self.replay_buffer = ReplayBuffer(10000)
        self.vis = True
        self.multiple = multiple

    def reset(self):
        self.episode = 0
        self.memory, self.action = self.M.copy(), self.A.copy()
        self.len_memory = len(self.memory)
        self.last_len = self.len_memory

        self.coefficient_matrix = np.array(self.memory).T
        self.set_memory = set([tuple(e) for e in self.memory])
        self.set_action = set([tuple(e) for e in self.action])
        self.tuple_memory = [tuple(e) for e in self.memory]
        self.last_gamma, _ = self.compute_linear_programming(self.len_memory, self.coefficient_matrix)
        self.gamma0 = abs(self.last_gamma)

        print('gamma0:', self.last_gamma)
        if self.last_gamma not in self.pos:
            self.pos[self.last_gamma] = 0

        self.state = (tuple(self.tuple_memory), [self.last_gamma, 0])

        self.map[tuple(self.tuple_memory)] = len(self.action)
        return self.state, True if self.last_gamma >= 0 else False

    def step(self, pos):
        action = self.action[pos]
        self.episode += 1

        self.add_memory(action)
        print(f'The steps:{self.episode}')

        if self.len_memory > self.last_len:
            gamma, _ = self.compute_linear_programming(self.len_memory, self.coefficient_matrix)
            self.map[tuple(self.tuple_memory)] = len(self.action)
        else:
            gamma = self.last_gamma
        self.last_len = self.len_memory

        done = True if gamma >= 0 else False

        if abs(gamma - self.last_gamma) < 1e-10:
            self.state = (tuple(self.tuple_memory), [gamma, self.state[1][1] + 1])
        else:
            self.state = (tuple(self.tuple_memory), [gamma, 0])
            if self.vis:
                self.handle_buffer(action, pos)

        reward = self.get_reward(gamma)

        truncated = True if self.episode >= self.max_episode else False

        print('state:', self.state[1], 'pos:', pos)
        print('reward:', reward, 'done:', done, 'len_memory:', self.len_memory, 'len_action:', len(self.action))

        return self.state, reward, done, truncated, self.episode

    def get_reward(self, gamma):
        reward = (gamma - self.last_gamma) / self.gamma0 * 10 - 0.1
        reward = round(reward, 2)
        self.last_gamma = gamma
        return reward

    def compute_linear_programming(self, len_memory, coefficient_matrix):
        x = cp.Variable((len_memory, 1))
        y = cp.Variable()
        no_constant = coefficient_matrix[1:]
        constant = coefficient_matrix[0:1]

        b = np.array([self.objective[1:]]).T
        A = np.diag(np.ones(len_memory))

        obj = cp.Maximize(y)
        zero = np.zeros((len_memory, 1))

        constraints = [A @ x >= zero, no_constant @ x == b, constant @ x == self.objective[0] - y]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)
        if prob.status == cp.OPTIMAL:
            s = coefficient_matrix @ x.value
            state = list(s.T[0])
            s = [[e[0]] if abs(e[0]) > 1e-6 else [0] for e in s]
            print('sum:', sum(self.sp_poly @ s))
            return round(float(y.value), 2), state
        else:
            return None, None

    def add_memory(self, memory):
        memory = list(memory)
        if tuple(memory) not in self.set_memory:
            self.set_memory.add(tuple(memory))
            self.tuple_memory.append(tuple(memory))
            self.memory.append(memory)
            self.len_memory += 1
            self.coefficient_matrix = np.concatenate((self.coefficient_matrix, np.array([memory]).T), axis=1)
            if self.M_deg_map[tuple(memory)] < self.l:
                tmp = []
                for mul in self.M_:
                    new_poly = mul_polynomial_with_fft(memory, mul, self.dic_forward, self.dic_reverse, self.len_vector,
                                                       self.max_map)
                    if tuple(new_poly) not in self.set_action:
                        self.set_action.add(tuple(new_poly))
                        tmp.append(new_poly)
                        self.M_deg_map[tuple(new_poly)] = self.M_deg_map[tuple(memory)] + 1
                if len(tmp) > 0:
                    self.action = np.concatenate((self.action, np.array(tmp)), axis=0)

    def add_action(self, action):
        memory = list(action)
        if self.M_deg_map[tuple(memory)] < self.l:
            tmp = []
            for mul in self.M_:
                new_poly = mul_polynomial_with_fft(memory, mul, self.dic_forward, self.dic_reverse, self.len_vector,
                                                   self.max_map)
                if tuple(new_poly) not in self.set_action:
                    self.set_action.add(tuple(new_poly))
                    tmp.append(new_poly)
                    self.M_deg_map[tuple(new_poly)] = self.M_deg_map[tuple(memory)] + 1
            if len(tmp) > 0:
                self.A = np.concatenate((self.A, np.array(tmp)), axis=0)

    def memory_initialization(self):
        max_obj_deg = self.deg
        M_ = []
        _, poly = monomials(self.n * 2, max_obj_deg)

        for i in range(self.n):
            tmp = [0] * self.len_vector
            tmp[i + 1] = 1
            M_.append(tmp)
        for i in range(self.n):
            tmp = [0] * self.len_vector
            tmp[0], tmp[i + 1] = 1, -1
            M_.append(tmp)
        self.M_ = M_
        poly = poly[1:]
        pool = Pool(processes=mp.cpu_count() // 3)
        res = pool.map(self.compute_memory, poly)
        pool.close()
        pool.join()

        for x, y in zip(res, poly):
            self.M_deg_map[tuple(x)] = sum(y)

        for i, x in enumerate(poly):
            if self.first_deg_pos < 0 and sum(x) == self.deg:
                self.first_deg_pos = i
                break

        self.M = res
        self.set_M = set([tuple(e) for e in self.M])
        self.origin_state = res

        self.memory_action = self.M[self.first_deg_pos:]
        action = []
        self.set_action = set()
        for item in self.memory_action:
            if self.M_deg_map[tuple(item)] < self.l:
                for mul in self.M_:
                    new_poly = mul_polynomial_with_fft(item, mul, self.dic_forward, self.dic_reverse, self.len_vector,
                                                       self.max_map)
                    if tuple(new_poly) not in self.set_action:
                        self.set_action.add(tuple(new_poly))
                        action.append(new_poly)
                        self.M_deg_map[tuple(new_poly)] = self.M_deg_map[tuple(item)] + 1
                        self.reverse[tuple(new_poly)] = (tuple(item), tuple(mul))
        self.A = np.array(action)

        print(f'Initial memory number:{len(res)}, Number of initial action sets:{len(self.A)}')
        self.M_copy, self.A_copy = self.M.copy(), self.A.copy()

    def compute_memory(self, item):
        res = [1] + [0] * (self.len_vector - 1)
        for k in range(len(item)):
            if item[k] > 0:
                for j in range(item[k]):
                    res = mul_polynomial_with_fft(res, self.M_[k], self.dic_forward, self.dic_reverse, self.len_vector,
                                                  self.max_map)
        return res

    def get_objective(self, item: dict):
        dic = {}
        for i, e in enumerate(self.poly):
            dic[e] = i
        res = [0] * self.len_vector
        for key, value in item.items():
            res[dic[key]] += value
        return res

    def handle_buffer(self, action, pos):
        len_memory = len(self.M)
        coefficient_matrix = np.array(self.M).T
        t1 = [tuple(e) for e in self.M]
        g1, _ = self.compute_linear_programming(len_memory, coefficient_matrix)

        self.M.append(list(action))  # np.concatenate((self.M, np.array([action])), axis=0)
        len_memory = len(self.M)
        coefficient_matrix = np.array(self.M).T
        t2 = [tuple(e) for e in self.M]
        g2, _ = self.compute_linear_programming(len_memory, coefficient_matrix)
        if g1 not in self.pos:
            self.pos[g1] = 0
        if g2 not in self.pos:
            self.pos[g2] = 0
        done = True if g2 >= 0 else False
        if abs(g2 - g1) < 1e-10:
            for i in range(self.multiple):
                self.replay_buffer.add((tuple(t1), [g1, self.pos[g1]]), pos, 0.2, (tuple(t2), [g1, self.pos[g1] + 1]),
                                       done)
            self.pos[g1] += 1
        else:
            for i in range(self.multiple):
                self.replay_buffer.add((tuple(t1), [g1, self.pos[g1]]), pos, 0.2, (tuple(t2), [g2, 0]), done)
        self.map[tuple(t1)] = len(self.action)
        self.map[tuple(t2)] = len(self.action)

    def update_parameters(self):
        self.vis = False
        self.M, self.A = self.M_copy.copy(), self.A_copy.copy()


if __name__ == '__main__':
    from proof.Example import get_examples_by_name

    ex = get_examples_by_name('C1')
    env = Env(ex, 500)
    print(env.objective)
    env.reset()
