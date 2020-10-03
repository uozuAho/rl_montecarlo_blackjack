import unittest

import ai_blackjack.blackjack.blackjack as bj
import ai_blackjack.monte_carlo_optimal_policy as mc_op


any_state = bj.State(0, 0, False)


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


class EmptyMutableAgent(unittest.TestCase):

    def setUp(self):
        self.agent = mc_op.MutableAgent()

    def test_any_action_raises_key_error(self):
        self.assertRaises(KeyError, lambda: self.agent.action(any_state))

    def test_returns_action_that_was_set(self):
        state = bj.State(0, 0, False)
        action = 0
        self.agent.set_action(state, action)

        self.assertEqual(self.agent.action(state), action)

    def test_has_no_actions(self):
        self.assertEqual(0, len(list(self.agent.all_actions())))


class EmptyActionValues(unittest.TestCase):

    def setUp(self):
        self.action_values = mc_op.ActionValues()

    def test_any_value_raises_key_error(self):
        self.assertRaises(KeyError, lambda: self.action_values.value(any_state, 0))

    def test_highest_value_action_raises_key_error(self):
        self.assertRaises(KeyError, lambda: self.action_values.highest_value_action(any_state))


class ActionValues(unittest.TestCase):

    def setUp(self):
        self.action_values = mc_op.ActionValues()

    def test_returns_action_value_that_was_set(self):
        state = bj.State(0, 0, False)
        action = 0
        value = 0.3
        self.action_values.set(state, action, value)

        self.assertEqual(self.action_values.value(state, action), value)

    def test_highest_value_action_has_highest_value(self):
        state = bj.State(0, 0, False)
        action_0 = 0
        action_1 = 1
        self.action_values.set(state, action_0, 0.1)
        self.action_values.set(state, action_1, 0.9)

        self.assertEqual(self.action_values.highest_value_action(state), action_1)


class ImprovePolicy(unittest.TestCase):

    def test_does_not_lower_policy_value(self):
        policy = mc_op.MutableAgent()
        action_values = mc_op.ActionValues()
        returns = mc_op.Returns()

        state = bj.State(0, 0, False)
        policy.set_action(state, 0)
        action_values.set(state, 0, 0.5)

        total_value_before = total_value(policy, action_values)

        mc_op.improve_policy(policy, action_values, returns)

        self.assertGreaterEqual(total_value(policy, action_values), total_value_before)


def total_value(policy, action_values: mc_op.ActionValues):
    return sum((action_values.value(s, a) for s, a in policy.all_actions()))
