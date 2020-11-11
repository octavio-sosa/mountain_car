import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# init q-value constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25_000 #debug?TODO

'''
print('observation space (high):', env.observation_space.high)
print('observation space (low):', env.observation_space.low)
print('action space:', env.action_space.n)
'''

# init observation space (discrete)
observ_bin_n = 20
observ_n = len(env.observation_space.high) #either high or low
observ_size = [observ_bin_n] * observ_n
observ_bin_size = (env.observation_space.high - env.observation_space.low)\
                 / observ_size

# init q_table 
q_table_size = observ_size + [env.action_space.n] #all observ combos for each action
q_table = np.random.uniform(low=-2, high=0, size=q_table_size)

state_current_i = get_discrete_state_index(env.reset())
done = False

while not done:
    action = np.argmax(q_table[state_current_i])
    state_new, reward, done, _ = env.step(action)
    state_new_i = get_discrete_state_index(state_new)

    env.render()
    if not done: 
        q_current, q_future_max, q_new = get_q_values(q_table, action, state_current_i, state_new_i)
        q_table[state_current_i + (action,)] = q_new

    elif state_new_i[0] >= env.goal_position: #reached goal
        q_table[state_current_i + (action,)] = 0 #max q-value
        
    state_current_i = state_new_i

env.close()

def get_discrete_state_index(state):
    index = (state - env.observation_space.low) / observ_bin_size
    return tuple(index.astype(np.int))

def get_q_values(q_table, action, state_current_i, state_new_i):

    q_future_max = np.max(q_table[state_new_i])
    q_current = q_table[state_current_i+ (action,)]
    q_new = (1-LEARNING_RATE)*q_current + LEARNING_RATE*(reward + DISCOUNT*q_future_max)
    return q_current, q_future_max, q_new
