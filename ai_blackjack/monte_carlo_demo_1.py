from typing import List, Tuple

def main():
    policy = StayOn20Agent()

class StayOn20Agent:
    def action(self, obs):
        return 1 if obs[0] < 20 else 0

class State:
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


class Episode:
    # steps: List(reward, State, action)
    def __init__(self, steps: List[Tuple[int, State, int]]):
        self._steps = steps
        self._states: List[State] = [step[1] for step in steps]

    def is_first_visit(self, state: State, t: int):
        return state not in self._states[0:t]

    def length(self):
        return len(self._steps)


def estimate_V(policy, gamma):
    # V = {s: arbitrary for s in all_states}
    # returns = {s: [] for s in all_states}

    # while True:
    #     G_return = 0
    #     episode = list(generate_episode)
    #     states = [ep[1] for ep in episode]
    #     rewards = [ep[0] for ep in episode]
    #     for t in reversed(range(len(episode) - 1)):
    #         state = states[t]
    #         G_return = gamma * G_return + rewards[t + 1]
    #         if state not in states[0:t]:  # <-- if first visit
    #             returns[state].append(G_return)
    #             V[state] = avg(returns[state])

    while True:
        G_return = 0
        episode = Episode(env, policy)
        for t in reversed(range(len(episode) - 1)):
            state = episode.state(t)
            G_return = gamma * G_return + episode.reward(t + 1)
            if episode.is_first_visit(state, t):
                returns[state].append(G_return)
                V[state] = avg(returns[state])

# episode: reward(for action in t-1), obs, action
def generate_episode(env, policy):
    obs = env.reset()
    done = False
    reward = None
    while not done:
        action = policy.action(obs)
        yield (reward, obs, action)
        obs, reward, done, _ = env.step(action)
    yield (reward, None, None)


if __name__ == "__main__":
    main()
