from typing import Iterable, Dict

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import ai_blackjack.blackjack.blackjack as bj


def print_values(values: Dict[bj.State, float]):
    _, _, z = _extract_xyz_from_values(values)
    np.set_printoptions(precision=2)
    print(z)


def plot_values(values: Dict[bj.State, float], title=None, has_usable_ace=False):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.gca(projection='3d')
    x, y, z = _extract_xyz_from_values(values, has_usable_ace)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def print_policy(policy):
    def _to_string(policy, usable_ace: bool):
        s = ""
        for p in range(21, 11, -1):
            line = f"{p} "
            for d in range(1, 11):
                state = bj.State(p, d, usable_ace)
                action = policy.action(state)
                action_str = "H" if action else "S"
                line += action_str + " "
            s += line + "\n"
        s += "   1 2 3 4 5 6 7 8 9 10\n"
        return s

    s = "no usable ace\n"
    s += "\n"
    s += _to_string(policy, False)
    s += "\n"
    s += "usable ace\n"
    s += "\n"
    s += _to_string(policy, True)
    print(s)


def _extract_xyz_from_values(values: Dict[bj.State, float], has_usable_ace=False):
    X = range(1, 11, 1)     # dealer showing
    Y = range(12, 22, 1)    # player sum
    z = []
    for y in Y:
        zrow = []
        z.append(zrow)
        for x in X:
            state = bj.State(y, x, has_usable_ace)
            value = values[state] if state in values else 0.0
            zrow.append(value)
    X, Y = np.meshgrid(X, Y)
    z = np.array(z)
    return X, Y, z
