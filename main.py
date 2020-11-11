import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

print('observation space (high):', env.observation_space.high)
print('observation space (low):', env.observation_space.low)
print('action space:', env.action_space.n)

done = False
action = 2

# init observation space (discrete)
observ_bin_n = 20
observ_n = len(env.observation_space.high) #either high or low
observ_size = [observ_bin_n] * observ_n
observ_bin_size = (env.observation_space.high - env.observation_space.low)\
                 / observ_size

# init q_table 
q_table_size = observ_size + [env.action_space.n] #all observ combos for each action
q_table = np.random.uniform(low=-2, high=0, size=q_table_size)

while not done:
    new_state, reward, done, _ = env.step(action)

    #print('reward:', reward, ', action:', action)
    env.render()

env.close()

def get_discrete_state_index(state, observ_space_low, observ_bin_size):
    index = (state - observ_space_low) / observ_bin_size
    return tuple(index.astype(np.int))
