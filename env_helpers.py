import numpy as np
import random

class QLearningApproximator:
    def __init__(self, num_features, num_actions, alpha=0.01, gamma=0.99, epsilon=0.1):
        """
        Q-learning with linear function approximation.

        Q(s, a) = w_a · φ(s)
        where φ(s) is the feature vector and w_a are the weights for action a.

        Args:
            num_features: Number of features representing the state.
            num_actions: Number of discrete possible actions.
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Exploration probability.
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize weights for each action (num_actions × num_features)
        self.weights = np.zeros((num_actions, num_features))

    def get_q_values(self, state_features):
        """Return Q-values for all actions given a state feature vector."""
        return np.dot(self.weights, state_features)

    def choose_action(self, state_features):
        """Epsilon-greedy policy for exploration/exploitation."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.get_q_values(state_features)
        return np.argmax(q_values)

    def update(self, state_features, action, reward, next_state_features, done):
        """
        Perform the Q-learning update step using linear approximation.

        Δw_a = α * (target - Q(s,a)) * φ(s)
        where target = r + γ * max_a' Q(s', a')
        """
        q_values = self.get_q_values(state_features)
        q_value = q_values[action]

        next_q_values = self.get_q_values(next_state_features)
        target = reward if done else reward + self.gamma * np.max(next_q_values)

        td_error = target - q_value
        self.weights[action] += self.alpha * td_error * state_features

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Gradually reduce exploration."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
