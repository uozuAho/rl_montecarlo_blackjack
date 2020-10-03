from typing import Iterable, Dict, Tuple

import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import ai_blackjack.blackjack.blackjack as bj
from ai_blackjack import visualise


def run_demo():
    policy, values = find_optimal_policy()
    visualise.print_policy(policy)
    visualise.plot_values(values,
        f'State values for optimal agent, no usable ace',
        False
    )
    # plot_values(values,
    #     f'State values for optimal agent, usable ace',
    #     True
    # )


class MutableAgent:
    def action(self, state):
        return 0

    def set_action(self, state, action):
        pass

    def all_actions(self) -> Iterable[Tuple[bj.State, int]]:
        yield (bj.State(0, 0, False), 0)


class ActionValues:
    def __init__(self):
        self._values: Dict[Tuple[bj.State, int], float] = {}

    def add(self, state: bj.State, action: int, value: float):
        self._values[(state, action)] = value

    def value(self, state: bj.State, action: int):
        return self._values[(state, action)]

    def highest_value_action(self, state: bj.State):
        return 0


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
    gamma = 1.0
    policy = MutableAgent()
    action_values = ActionValues()
    returns = Returns()

    for _ in range(10):
        state = bj.State(0, 0, False) # todo: random state
        action = 0 # todo: random action
        G_return = 0
        # todo: generate episode with given starting state and action (exploring start)
        episode = bj.Episode(list(generate_episode(state, action)))
        for t in reversed(range(episode.length() - 1)):
            state = episode.steps[t].state
            action = episode.steps[t].action
            G_return = gamma * G_return + episode.steps[t + 1].reward
            if episode.first_visit(state, action) == t: # todo: is first visit for state and action
                returns.add(state, action, G_return)
                action_values.add(state, action, returns.average_for(state, action))
                best_action = action_values.highest_value_action(state)
                policy.set_action(state, best_action)

    return policy, {bj.State(0, 0, False): 0.0}


def improve_policy(policy: MutableAgent, action_values: ActionValues, returns: Returns) -> None:
    state = bj.State(0, 0, False) # todo: random state
    action = 0 # todo: random action
    G_return = 0
    # todo: generate episode with given starting state and action (exploring start)
    episode = bj.Episode(list(generate_episode(state, action)))
    for t in reversed(range(episode.length() - 1)):
        state = episode.steps[t].state
        action = episode.steps[t].action
        G_return = G_return + episode.steps[t + 1].reward
        if episode.first_visit(state, action) == t: # todo: is first visit for state and action
            returns.add(state, action, G_return)
            action_values.add(state, action, returns.average_for(state, action))
            best_action = action_values.highest_value_action(state)
            policy.set_action(state, best_action)


def generate_episode(first_state, first_action):
    yield bj.EpisodeStep(0, bj.State(0, 0, False), 0)


def avg(things):
    list_of_things = list(things)
    return sum(list_of_things) / len(list_of_things)
