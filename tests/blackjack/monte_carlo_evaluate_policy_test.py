import unittest

import ai_blackjack.monte_carlo_evaluate_policy as mc1


class AlwaysStayAgent:
    def action(self, observation):
        return 0


class E2eTests(unittest.TestCase):

    def test_values_is_a_non_empty_dict_of_states(self):
        values = mc1.estimate_V(AlwaysStayAgent(), 10)

        self.assertIsInstance(values, dict)
        self.assertTrue(len(values) > 0)
        for k, _ in values.items():
            self.assertIsInstance(k, mc1.State)


# todo: how to run this case multiple times?
class Generated_Episode_When_Player_Always_Stays(unittest.TestCase):

    def setUp(self):
        self.episode = list(mc1.generate_episode(AlwaysStayAgent()))

    def test_is_never_empty(self):
        self.assertTrue(len(self.episode) > 0)

    def test_contains_episode_steps(self):
        for step in self.episode:
            self.assertIsInstance(step, mc1.EpisodeStep)

    def test_episode_step_types(self):  # boo to dynamic types!
        for step in self.episode:
            self.assertTrue(isinstance(step.reward, float) or step.reward is None)
            self.assertIsInstance(step.state, mc1.State)
            self.assertTrue(isinstance(step.action, int) or step.action is None)

    def test_first_step_has_no_reward(self):
        self.assertIsNone(self.episode[0].reward)

    def test_last_step_has_no_action(self):
        self.assertIsNone(self.episode[-1].action)

    def test_player_never_has_over_21(self):
        last_state = self.episode[-1].state
        self.assertTrue(last_state.player_sum <= 21)
