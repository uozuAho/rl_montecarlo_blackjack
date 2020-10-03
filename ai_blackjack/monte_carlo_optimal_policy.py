from typing import Iterable, Dict, Tuple
import random

import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import ai_blackjack.blackjack.blackjack as bj
from ai_blackjack import visualise


def run_demo():
    policy, values = find_optimal_policy()
    visualise.print_policy(policy)
    visualise.print_values(values)
    visualise.plot_values(values,
        f'State values for optimal policy, no usable ace',
        False
    )
    # plot_values(values,
    #     f'State values for optimal policy, usable ace',
    #     True
    # )


class MutablePolicy:
    def __init__(self):
        self._actions: Dict[bj.State, int] = {}

    def action(self, state):
        if state not in self._actions:
            return 0
        return self._actions[state]

    def set_action(self, state, action):
        self._actions[state] = action

    def all_actions(self) -> Iterable[Tuple[bj.State, int]]:
        for state, action in self._actions.items():
            yield state, action


class ExploringStartPolicy:
    def __init__(self, policy, first_action):
        self.policy = policy
        self.first_action = first_action
        self.is_first_action = True

    def action(self, state):
        if self.is_first_action:
            self.is_first_action = False
            return self.first_action
        else:
            return self.policy.action(state)

    def set_action(self, state, action):
        return self.policy.set_action(state, action)

    def all_actions(self) -> Iterable[Tuple[bj.State, int]]:
        for state, action in self.policy.all_actions():
            yield state, action


class ActionValues:
    def __init__(self):
        # { state: { action: value } }
        self._values = {}

    def set(self, state: bj.State, action: int, value: float):
        if state not in self._values:
            self._values[state] = { action: value }
        else:
            self._values[state][action] = value

    def value(self, state: bj.State, action: int):
        return self._values[state][action]

    def highest_value_action(self, state: bj.State):
        values = self._values[state]
        return max(values.items(), key=lambda v: v[1])[0]


class Returns:
    def __init__(self):
        self._returns: Dict[Tuple[bj.State, int], float] = {}

    def add(self, state: bj.State, action: int, return_: float):
        state_action = (state, action)
        if state_action in self._returns:
            self._returns[state_action].append(return_)
        else:
            self._returns[state_action] = [return_]

    def average_for(self, state: bj.State, action: int):
        return avg(self._returns[(state, action)])


def find_optimal_policy():
    policy = MutablePolicy()
    action_values = ActionValues()
    returns = Returns()

    for _ in range(100):
        improve_policy(policy, action_values, returns)

    return policy, { state: action_values.value(state, action) for state, action in policy.all_actions() }


def improve_policy(policy: MutablePolicy, action_values: ActionValues, returns: Returns) -> None:
    reward_sum = 0
    first_action = random.randint(0, 1)
    exploring_policy = ExploringStartPolicy(policy, first_action)
    # note: we get exploring starting states for free with the random episode generator
    episode = bj.Episode(list(bj.generate_random_episode(exploring_policy)))
    for t in reversed(range(episode.length() - 1)):
        state = episode.steps[t].state
        action = episode.steps[t].action
        reward_sum += episode.steps[t + 1].reward
        if episode.first_visit(state, action) == t:
            returns.add(state, action, reward_sum)
            action_values.set(state, action, returns.average_for(state, action))
            best_action = action_values.highest_value_action(state)
            policy.set_action(state, best_action)


def avg(things):
    list_of_things = list(things)
    return sum(list_of_things) / len(list_of_things)
