import unittest

import ai_blackjack.blackjack.blackjack as blackjack


class GivenEpisodeOfDuplicateStates(unittest.TestCase):

    def setUp(self):
        state = blackjack.State(10, 10, False)
        self.state = state
        self.episode = blackjack.Episode([
            blackjack.EpisodeStep(None, blackjack.State.copy(state), 0),
            blackjack.EpisodeStep(None, blackjack.State.copy(state), 0),
        ])

    def test_first_visit_is_the_first_state(self):
        self.assertEqual(self.episode.first_visit(self.state), 0)


class GivenEpisodeOfDifferentStates(unittest.TestCase):

    def setUp(self):
        self.steps = [
            blackjack.EpisodeStep(0, blackjack.State(0, 0, False), 0),
            blackjack.EpisodeStep(1, blackjack.State(1, 1, True), 1)
        ]
        self.episode = blackjack.Episode(self.steps)

    def test_every_state_is_first_visit(self):
        for t in range(1, self.episode.length()):
            with self.subTest(t):
                state = self.steps[t].state
                self.assertEqual(self.episode.first_visit(state), t)
