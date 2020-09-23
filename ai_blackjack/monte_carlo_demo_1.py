from typing import List, Tuple, Iterable, Dict

import gym


def main():
    env = gym.make('Blackjack-v0')
    policy = StayOn20Agent()
    values = estimate_V(env, policy)
    for state in values:
        print(state, values[state])


class StayOn20Agent:
    def action(self, obs):
        return 1 if obs[0] < 20 else 0


class State:
    """ The state of a blackjack game
        Attributes:
            player_sum:     sum of the player's cards
            dealers_card:   the dealer's card that the player can see
            has_usable_ace: True if the player has a 'usable' ace
    """
    def __init__(self, player_sum: int, dealers_card: int, has_usable_ace: bool):
        self.player_sum = player_sum
        self.dealers_card = dealers_card
        self.has_usable_ace = has_usable_ace

    @classmethod
    def from_obs(cls, obs):
        return cls(obs[0], obs[1], obs[2])

    @classmethod
    def copy(cls, s):
        return cls(s.player_sum, s.dealers_card, s.has_usable_ace)

    def __eq__(self, other):
        return isinstance(other, State) and self.__dict__.items() == other.__dict__.items()

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class EpisodeStep:
    """ A step of an episode (game) of blackjack

        Attributes:
            reward: the reward for the previous action, resulting in this state
            state:  the current state of the game
            action: the action taken from this state
    """
    def __init__(self, reward: int, state: State, action: int):
        self.reward = reward
        self.state = state
        self.action = action


class Episode:
    def __init__(self, steps: List[EpisodeStep]):
        self.steps = steps
        self._states: List[State] = [step.state for step in steps]

    def is_first_visit(self, state: State, t: int):
        return state not in self._states[0:t]

    def length(self):
        return len(self.steps)


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
