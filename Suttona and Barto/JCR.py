import numpy as np
from scipy.stats import poisson

class JCR:

    def __init__(self, cost=2, reward=10, lambda_req1=3, lambda_req2=4, max_cars_each=20, lambda_ret1=3, lambda_ret2=2, theta=1e-6, gamma=0.9):
        self.lambda_req1 = lambda_req1
        self.lambda_req2 = lambda_req2
        self.lambda_ret1 = lambda_ret1
        self.lambda_ret2 = lambda_ret2
        self.max_cars_each = max_cars_each
        self.theta = theta
        self.cost = cost
        self.reward = reward
        self.gamma = gamma
        self.poisson_cache = {}

# We cache the Poisson probabilities to avoid recalculating them to save time
    def poisson_prob(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]

    def expected_return(self, state, action, statevalue):
        returns = -self.cost * abs(action)
        cars_start1 = int(min(state[0] - action, self.max_cars_each))
        cars_start2 = int(min(state[1] + action, self.max_cars_each))

        for rent1 in range(11):
            for rent2 in range(11):
                prob_rent = self.poisson_prob(rent1, self.lambda_req1) * self.poisson_prob(rent2, self.lambda_req2)

                real_rent1 = min(cars_start1, rent1)
                real_rent2 = min(cars_start2, rent2)
                reward = (real_rent1 + real_rent2) * self.reward

                cars_left1 = cars_start1 - real_rent1
                cars_left2 = cars_start2 - real_rent2

                expected_value = 0.0
                for ret1 in range(11):
                    for ret2 in range(11):
                        prob_ret = self.poisson_prob(ret1, self.lambda_ret1) * self.poisson_prob(ret2, self.lambda_ret2)

                        cars_end1 = min(cars_left1 + ret1, self.max_cars_each)
                        cars_end2 = min(cars_left2 + ret2, self.max_cars_each)

                        prob = prob_rent * prob_ret
                        expected_value += prob * statevalue[cars_end1, cars_end2]

                returns += prob_rent * (reward + self.gamma * expected_value)

        return returns

    def policy_iteration(self):
        state_value = np.zeros((21, 21))
        policy = np.zeros((21, 21), dtype=int)

        stable_policy = False
        actions = np.arange(-5, 6)

        while not stable_policy:
            # Policy Evaluation
            while True:
                delta = 0
                for i in range(21):
                    for j in range(21):
                        v = state_value[i, j]
                        action = policy[i, j]
                        state_value[i, j] = self.expected_return([i, j], action, state_value)
                        delta = max(delta, abs(v - state_value[i, j]))
                if delta < self.theta:
                    break

            # Policy Improvement
            stable_policy = True
            for i in range(21):
                for j in range(21):
                    old_action = policy[i, j]
                    action_returns = []

                    for action in actions:
                        if 0 <= i - action <= self.max_cars_each and 0 <= j + action <= self.max_cars_each:
                            val = self.expected_return([i, j], action, state_value)
                            action_returns.append((val, action))

                    best_action = max(action_returns)[1]
                    policy[i, j] = best_action

                    if old_action != best_action:
                        stable_policy = False

        return policy, state_value
