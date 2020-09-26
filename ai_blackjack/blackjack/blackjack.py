from typing import List, Tuple, Iterable, Dict


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

    def __str__(self):
        return f"p: {self.player_sum}, d: {self.dealers_card}, a: {self.has_usable_ace}"

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
    def __init__(self, reward: float, state: State, action: int):
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
