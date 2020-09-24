from typing import Iterable, Dict

import gym

from blackjack.blackjack import Episode, EpisodeStep, State


def main():
    env = gym.make('Blackjack-v0')
    policy = StayOn20Agent()
    values = estimate_V(env, policy)
    print_values(values)


def print_values(values: Dict[State, float]):
    for state in values:
        print(state, values[state])


class StayOn20Agent:
    def action(self, obs):
        return 1 if obs[0] < 20 else 0


def estimate_V(env, policy) -> Dict[State, float]:
    gamma = 1
    returns = {}

    for _ in range(10):
    # while True:  # todo: stop when converged
        G_return = 0
        episode = Episode(list(generate_episode(env, policy)))
        for t in reversed(range(episode.length() - 1)):
            state = episode.steps[t].state
            G_return = gamma * G_return + episode.steps[t + 1].reward
            if episode.is_first_visit(state, t):
                if state in returns:
                    returns[state].append(G_return)
                else:
                    returns[state] = [G_return]

    return {s: sum(returns[s]) / len(returns[s]) for s in returns.keys()}


def generate_episode(env, policy) -> Iterable[EpisodeStep]:
    obs = env.reset()
    done = False
    reward = None
    while not done:
        action = policy.action(obs)
        yield EpisodeStep(reward, obs, action)
        obs, reward, done, _ = env.step(action)
    yield EpisodeStep(reward, None, None)


if __name__ == "__main__":
    main()
