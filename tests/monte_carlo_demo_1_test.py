import unittest

import ai_blackjack.monte_carlo_demo_1 as mc

class GivenEpisodeOfDuplicateStates(unittest.TestCase):

    def setUp(self):
        state = mc.State(10, 10, False)
        self.state = state
        self.episode = mc.Episode([
            mc.EpisodeStep(None, mc.State.copy(state), 0),
            mc.EpisodeStep(None, mc.State.copy(state), 0),
        ])

    def test_first_visit_is_the_first_state(self):
        is_first_visit = self.episode.is_first_visit(self.state, 0)
        self.assertTrue(is_first_visit)

    def test_no_other_state_is_first_visit(self):
        for t in range(1, self.episode.length()):
            with self.subTest(t):
                is_first_visit = self.episode.is_first_visit(self.state, t)
                self.assertFalse(is_first_visit)


class GivenEpisodeOfDifferentStates(unittest.TestCase):

    def setUp(self):
        self.steps = [
            mc.EpisodeStep(0, mc.State(0, 0, False), 0),
            mc.EpisodeStep(1, mc.State(1, 1, True), 1)
        ]
        self.episode = mc.Episode(self.steps)

    def test_every_state_is_first_visit(self):
        for t in range(1, self.episode.length()):
            with self.subTest(t):
                state = self.steps[t].state
                is_first_visit = self.episode.is_first_visit(state, t)
                self.assertTrue(is_first_visit)