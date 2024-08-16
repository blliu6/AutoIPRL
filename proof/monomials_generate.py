import re
from functools import reduce

import sympy as sp


def dfs(n: int, m: int):
    now = [0] * n
    while True:
        yield now
        now[0] += 1
        for i in range(n):
            if now[i] > m or sum(now) > m:
                now[i] = 0
                if i + 1 >= n:
                    return
                now[i + 1] += 1
            else:
                break


def key_sort(item: list):
    res = [sum(item)]
    for i in range(len(item)):
        res.append(-item[i])
    return tuple(res)


def monomials(variable_number, degree):
    poly = [x.copy() for x in dfs(variable_number, degree)]
    poly.sort(key=key_sort)
    x = sp.symbols([f'x{i + 1}' for i in range(variable_number)])
    polynomial = [str(reduce(lambda a, b: a * b, [x[i] ** exp for i, exp in enumerate(e)])) for e in poly]
    return polynomial, poly


def replacer(match):
    return 'x' + str(int(match.group()[1:]) + 1)


def tran(s: str):
    s = s.replace(' ', '*')
    s = s.replace('^', '**')
    s = re.sub(r'x\d+', replacer, s)
    return s
