from typing import Iterable, Dict

import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from ai_blackjack.blackjack.blackjack import Episode, EpisodeStep, State


def run_demo():
    policy = StayOn20Agent()
    values = estimate_V(policy)
    # print_values(values)
    plot_values(values)


def print_values(values: Dict[State, float]):
    # for state in values:
    #     print(state, values[state])
    _, _, z = extract_xyz_from_values(values)
    np.set_printoptions(precision=2)
    print(z)


def plot_values(values: Dict[State, float]):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = extract_xyz_from_values(values)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def extract_xyz_from_values(values: Dict[State, float]):
    X = range(1, 11, 1)     # dealer showing
    Y = range(12, 22, 1)    # player sum
    z = []
    for x in X:
        zrow = []
        z.append(zrow)
        for y in Y:
            state = State(y, x, False)
            value = values[state] if state in values else 0.0
            zrow.append(value)
    z = np.array(z)
    return X, Y, z


class StayOn20Agent:
    def action(self, obs):
        return 1 if obs[0] < 20 else 0


def estimate_V(policy, episode_limit=10000) -> Dict[State, float]:
    env = gym.make('Blackjack-v0')

    gamma = 1
    returns = {}

    for _ in range(episode_limit):
    # while True:  # todo: stop when converged
        G_return = 0
        episode = Episode(list(generate_episode(policy, env)))
        for t in reversed(range(episode.length() - 1)):
            state = episode.steps[t].state
            G_return = gamma * G_return + episode.steps[t + 1].reward
            if episode.is_first_visit(state, t):
                if state in returns:
                    returns[state].append(G_return)
                else:
                    returns[state] = [G_return]

    return {s: sum(returns[s]) / len(returns[s]) for s in returns.keys()}


def generate_episode(policy, env=None) -> Iterable[EpisodeStep]:
    if not env: env = gym.make('Blackjack-v0')
    obs = env.reset()
    done = False
    reward = None
    while not done:
        action = policy.action(obs)
        yield EpisodeStep(reward, State.from_obs(obs), action)
        obs, reward, done, _ = env.step(action)
    yield EpisodeStep(reward, State.from_obs(obs), None)


if __name__ == "__main__":
    run_demo()
