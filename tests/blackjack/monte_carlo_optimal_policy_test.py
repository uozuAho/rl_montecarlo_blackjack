import unittest

import ai_blackjack.blackjack.blackjack as bj
import ai_blackjack.monte_carlo_optimal_policy as mc_op


class AlwaysStayAgent:
    def action(self, observation):
        return 0


class FindOptimalPolicy(unittest.TestCase):

    def setUp(self):
        self.policy, self.values = mc_op.find_optimal_policy()

    def test_returns_policy_and_values(self):
        self.assertTrue(hasattr(self.policy, 'action'))
        self.assertIsInstance(self.values, dict)

    def test_values_is_a_non_empty_dict_of_states(self):
        self.assertTrue(len(self.values) > 0)
        for k, _ in self.values.items():
            self.assertIsInstance(k, bj.State)
