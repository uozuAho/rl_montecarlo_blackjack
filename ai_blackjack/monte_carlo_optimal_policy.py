from typing import Iterable, Dict, Tuple

import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from ai_blackjack.blackjack.blackjack import Episode, EpisodeStep, State


def run_demo():
    policy = find_optimal_policy()


class MutableAgent:
    def action(self, obs):
        return 0

    def set_action(self, state, action):
        pass


def find_best_action(action_values, state):
    return 0


def generate_episode(first_state, first_action):
    yield EpisodeStep(0, State(0, 0, False), 0)


def avg(things):
    list_of_things = list(things)
    return sum(list_of_things) / len(list_of_things)


def find_optimal_policy():
    gamma = 1.0
    policy = MutableAgent()

    action_values: Dict[Tuple[State, int], float] = {}
    returns:       Dict[Tuple[State, int], float] = {}

    while False:
        state = State(0, 0, False) # todo: random state
        action = 0 # todo: random action
        G_return = 0
        episode = Episode(list(generate_episode(state, action))) # todo: generate episode
        for t in reversed(range(episode.length() - 1)):
            state = episode.steps[t].state
            action = episode.steps[t].action
            G_return = gamma * G_return + episode.steps[t + 1].reward
            if episode.is_first_visit(state, t): # todo: is first visit for state and action
                state_action = (state, action)
                if state_action in returns:
                    returns[state_action].append(G_return)
                else:
                    returns[state_action] = [G_return]

                action_values[state_action] = avg(returns[state_action])
                best_action = find_best_action(action_values, state)
                policy.set_action(state, best_action)

    return policy


if __name__ == "__main__":
    run_demo()
