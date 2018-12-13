import unittest
import PolicyValueIteration as cls
import gym
import numpy as np

class PolicyValueIterationTests(unittest.TestCase):
    def setUp(self):
        self.environment = gym.make('FrozenLake-v0').env
        self.value = [0., 0., 0., 0., 0., 0., 0., 0., 0.0663, 0.133, 0.266, 0., 0., 0.333, 0.666, 0.]
        self.policy = np.zeros(self.environment.nS)
        self.environment.reset()

    def test_execute_returns_total_reward(self):
        self.assertGreaterEqual(cls.get_reward(self.environment, self.policy), 0.0)

    def test_get_value_function_returns_value_greater_than_zero(self):
        self.policy = np.random.choice(self.environment.nA, self.environment.nS)
        result = cls.get_value_function(self.environment, self.policy)
        print(result)
        self.assertGreater(np.sum(result), 0.0)

    def test_get_policy_return_valid_policy(self):
        result = cls.get_policy(self.environment, self.value)
        print(result)
        self.assertGreater(np.sum(result), 0.0)

    def test_value_iteration_returns_valid_value(self):
        result = cls.value_iteration(self.environment, 1000)
        print(result)
        self.assertGreater(np.sum(result), 0.0)


if __name__ == '__main__':
    unittest.main()
