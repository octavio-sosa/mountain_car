import numpy as np

def make_table(num_val_per_observation, len_observation_space, num_actions, reward_range):
    observation_space_shape = [num_val_per_observation] * len_observation_space
    shape = observation_space_shape + [num_actions]
    neutral_reward = (reward_range[1] - reward_range[0]) / 2

    return np.full(shape, neutral_reward)

class agent: 
    def __init__(self, LEARNING_RATE, DISCOUNT,
                 observation_space, num_val_per_observation,
                 action_cost):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.observation_space = observation_space
        self.observation_size = (self.observation_space.high - self.observation_space.low)\
                                / num_val_per_observation
        self.action_cost = action_cost
        self.table = np.zeros(0)
        self.current_state_i = 0
        self.new_state_i = 0
        self.action = 0
        self.reward = 0
        self.q_value = 0;

    def update_table(self, done, position):
        future_max_q = np.max(self.table[self.new_state_i])
        self.q_value = self.table[self.current_state_i + (self.action,)]

        if done and position >= 0.5: #achieved goal
            #self.reward = 0 #max reward
            self.table[self.current_state_i + (self.action,)] = 0
        else:
            self.q_value += self.LEARNING_RATE*(self.reward + self.DISCOUNT*future_max_q \
                                                - self.q_value - self.action_cost)
            self.table[self.current_state_i + (self.action,)] = self.q_value

    def update_state_index(self, _type, state):
        index = (state - self.observation_space.low) / self.observation_size
        if _type == 'new':
            self.new_state_i = tuple(index.astype(np.int))
        elif _type == 'current':
            self.current_state_i = tuple(index.astype(np.int))


    def update_action(self, _type):
        if _type == 'table':
            self.action = np.argmax(self.table[self.current_state_i])
        elif _type == 'random':
            self.action = np.random.randint(0,3)
