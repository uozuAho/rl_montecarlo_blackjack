import unittest

import ai_blackjack.monte_carlo_demo_1 as mc

class GivenEpisodeOfDuplicateStates(unittest.TestCase):

    def setUp(self):
        state = mc.State(10, 10, False)
        self.state = state
        self.episode = mc.Episode([
            (None, mc.State.copy(state)),
            (None, mc.State.copy(state)),
        ])

    def test_first_visit_is_the_first_state(self):
        is_first_visit = self.episode.is_first_visit(self.state, 0)
        self.assertTrue(is_first_visit)

    def test_no_other_state_is_first_visit(self):
        for t in range(1, self.episode.length()):
            with self.subTest(t):
                is_first_visit = self.episode.is_first_visit(self.state, t)
                self.assertFalse(is_first_visit)