import sympy as sp


class Example:
    def __init__(self, n, obj_deg, objective, l, name):
        self.n = n
        self.obj_deg = obj_deg
        self.objective = objective
        self.l = l
        self.name = name


def convert(origin: str, bounds):
    for i, bound in enumerate(bounds):
        low, high = bound[0], bound[1]
        origin = origin.replace(f'x{i + 1}', f'({high - low}*x{i + 1}+{low})')
    origin = sp.sympify(origin)
    origin = sp.expand(origin)
    print(origin)
    terms = origin.as_ordered_terms()
    dic = {}
    for term in terms:
        coef = term.as_coeff_Mul()
        dic[str(coef[1])] = coef[0]
    if '1' not in dic:
        dic['1'] = 0
    return dic


examples = {
    1: Example(
        n=2,
        obj_deg=2,
        objective=('2*x2**2-x1+1.67', [[-1, 1]] * 2),
        l=3,
        name='C1'
    ),
    2: Example(
        n=2,
        obj_deg=2,
        objective=('x1**2+4*x2**2+1.67', [[-1, 1]] * 2),
        l=3,
        name='C2'
    ),
    3: Example(
        n=3,
        obj_deg=2,
        objective=('-(x1-2*x2+x3+0.835634534*x2*(1-x2))+18.904230228', [[-5, 5]] * 3),
        l=3,
        name='C3'
    ),
    4: Example(
        n=3,
        obj_deg=3,
        objective=('x1*x2**2+x1*x3**2-1.1*x1+10.35', [[-1.5, 2]] * 3),
        l=4,
        name='C4'
    ),
    5: Example(
        n=4,
        obj_deg=3,
        objective=('x1*x2**2+x1*x3**2+x1*x4**2-1.1*x1+21.8', [[-2, 2]] * 4),
        l=4,
        name='C5'
    ),
    6: Example(
        n=4,
        obj_deg=3,
        objective=('-(x1*x2**2+x1*x3**2+x1*x4**2-1.1*x1+1)+22.8', [[-2, 2]] * 4),
        l=4,
        name='C6'
    ),
    7: Example(
        n=4,
        obj_deg=4,
        objective=(
            '-(-x1*x3**3+4*x2*x3**2*x4+4*x1*x3*x4**2+2*x2*x4**3+4*x1*x3+4*x3**2-10*x2*x4-10*x4**2+2)+5327',
            [[-5, 5]] * 4),
        l=5,
        name='C7'
    ),
    8: Example(
        n=5,
        obj_deg=2,
        objective=('x5**2+x1+x2+x3+x4-x5+30', [[-5, 5]] * 5),
        l=3,
        name='C8'
    ),
    9: Example(
        n=5,
        obj_deg=4,
        objective=('x1*x2*x3*x4+x1*x2*x3*x5+x1*x2*x4*x5+x1*x3*x4*x5+x2*x3*x4*x5+30000', [[-10, 10]] * 5),
        l=5,
        name='C9'
    ),
    10: Example(
        n=6,
        obj_deg=2,
        objective=(
            '-6.8*x1*x4-3.2*x1*x5+1.3*x1*x6+5.1*x1-3.2*x3*x4-4.8*x2*x5-0.7*x2*x6-7.1*x2+1.3*x3*x4-0.7*x3*x5+9*x3*x6'
            '-x3+5.1*x4-7.1*x5-x6+270400', [[-100, 100]] * 6),
        l=3,
        name='C10'
    )
}


def get_examples_by_id(identify):
    return examples[identify]


def get_examples_by_name(name):
    for key in examples.keys():
        if examples[key].name == name:
            example = examples[key]
            example.objective = convert(*example.objective)
            return example


if __name__ == '__main__':
    ex = get_examples_by_name('case_4')
    print(ex.objective)
