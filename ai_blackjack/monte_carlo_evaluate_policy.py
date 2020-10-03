from typing import Iterable, Dict

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import ai_blackjack.blackjack.blackjack as bj
from ai_blackjack import visualise


def run_demo():
    policy = StayOn20Policy()
    num_episodes = 50000
    values = estimate_V(policy, num_episodes)
    visualise.plot_values(values,
        f'State values for stay on 20/21 policy, no usable ace, {num_episodes} episodes',
        False
    )
    visualise.plot_values(values,
        f'State values for stay on 20/21 policy, usable ace, {num_episodes} episodes',
        True
    )
    visualise.print_values(values)


class StayOn20Policy:
    def action(self, obs):
        return 1 if obs[0] < 20 else 0


def estimate_V(policy, episode_limit=10000) -> Dict[bj.State, float]:
    gamma = 1
    returns = {}

    for _ in range(episode_limit):
        G_return = 0
        episode = bj.Episode(list(bj.generate_random_episode(policy)))
        for t in reversed(range(episode.length() - 1)):
            state = episode.steps[t].state
            G_return = gamma * G_return + episode.steps[t + 1].reward
            if episode.first_visit(state) == t:
                if state in returns:
                    returns[state].append(G_return)
                else:
                    returns[state] = [G_return]

    # average all returns for each state
    return {s: sum(returns[s]) / len(returns[s]) for s in returns.keys()}


if __name__ == "__main__":
    run_demo()
