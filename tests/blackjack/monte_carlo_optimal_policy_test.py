import unittest

import ai_blackjack.monte_carlo_optimal_policy as mc_op


class AlwaysStayAgent:
    def action(self, observation):
        return 0


class FindOptimalPolicy(unittest.TestCase):

    def test_returns_policy(self):
        policy = mc_op.find_optimal_policy()

        self.assertTrue(hasattr(policy, 'action'))
