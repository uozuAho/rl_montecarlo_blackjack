import unittest

import ai_blackjack.blackjack.blackjack as blackjack


class AnyEpisode(unittest.TestCase):
    def setUp(self):
        self.episode = blackjack.Episode([
            blackjack.EpisodeStep(None, blackjack.State(0, 0, False), 0),
        ])

    def test_throws_when_given_state_is_not_in_episode(self):
        absent_state = blackjack.State(9, 9, True)
        self.assertRaises(ValueError, lambda: self.episode.first_visit(absent_state))

    def test_throws_when_state_action_pair_not_in_episode(self):
        present_state = blackjack.State(0, 0, False)
        absent_action = 1
        self.assertRaises(Exception, lambda: self.episode.first_visit(present_state, absent_action))


class GivenEpisodeOfDuplicateStateActions(unittest.TestCase):

    def setUp(self):
        state = blackjack.State(10, 10, False)
        self.state = state
        self.action = 0
        self.episode = blackjack.Episode([
            blackjack.EpisodeStep(None, blackjack.State.copy(state), self.action),
            blackjack.EpisodeStep(None, blackjack.State.copy(state), self.action),
        ])

    def test_first_visit_is_the_first_state(self):
        self.assertEqual(self.episode.first_visit(self.state), 0)

    def test_first_visit_is_the_first_state_and_action(self):
        self.assertEqual(self.episode.first_visit(self.state, self.action), 0)


class GivenEpisodeOfDifferentStateActions(unittest.TestCase):

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

    def test_every_state_action_is_first_visit(self):
        for t in range(1, self.episode.length()):
            with self.subTest(t):
                state = self.steps[t].state
                action = self.steps[t].action
                self.assertEqual(self.episode.first_visit(state, action), t)
