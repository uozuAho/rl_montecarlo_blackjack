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


class ImprovePolicy(unittest.TestCase):

    def test_improves_policy(self):
        policy = mc_op.MutableAgent()
        action_values = mc_op.ActionValues()
        returns = mc_op.Returns()

        state = bj.State(0, 0, False)
        policy.set_action(state, 0)
        action_values.add(state, 0, 0.5)

        total_value_before = total_value(policy, action_values)

        mc_op.improve_policy(policy, action_values, returns)

        self.assertGreater(total_value(policy, action_values), total_value_before)


def total_value(policy, action_values: mc_op.ActionValues):
    return sum((action_values.value(s, a) for s, a in policy.all_actions()))
