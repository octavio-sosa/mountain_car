import gym
import numpy as np
from plot.plot import plot

def get_state_index(state, observ_space_low, observ_bin_size):
    index = (state - observ_space_low) / observ_bin_size
    return tuple(index.astype(np.int))

def main():
    env = gym.make("MountainCar-v0")
    env.reset()

    # init q-value constants
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95

    EPOCHS = 20_000
    PERIOD = 1000

    epsilon = 0.5
    EPOCH_END_DECAY = EPOCHS//2
    EPSILON_DECAY = epsilon/EPOCH_END_DECAY

    # init observation space (discrete)
    observ_bin_n = 20
    observ_n = len(env.observation_space.high) #either high or low
    observ_size = [observ_bin_n] * observ_n
    observ_bin_size = (env.observation_space.high - env.observation_space.low)\
                     / observ_size

    # init q_table 
    q_table_size = observ_size + [env.action_space.n] #all observ combos for each action
    q_table = np.random.uniform(low=-2, high=0, size=q_table_size)
    
    progress = plot(EPOCHS, 100)

    for epoch in range(EPOCHS):
        if (epoch % PERIOD == 0) or (epoch == EPOCHS-1):
            print('epoch:', epoch)
            period_new = True
        else: 
            period_new = False

        state_current_i = get_state_index(env.reset(), env.observation_space.low, observ_bin_size)
        done = False
        steps = 0
        total_rewards = 0
        while not done:
            '''
            if period_new:
                env.render()
            '''

            if np.random.random() > epsilon:
                action = np.argmax(q_table[state_current_i])
            else:
                action = np.random.randint(0, env.action_space.n)

            state_new, reward, done, _ = env.step(action)
            state_new_i = get_state_index(state_new, env.observation_space.low, observ_bin_size)

            if not done: 
                #compute q-values
                q_future_max = np.max(q_table[state_new_i])
                q_current = q_table[state_current_i+ (action,)]
                q_new = (1-LEARNING_RATE)*q_current + LEARNING_RATE*(reward + DISCOUNT*q_future_max)
                q_table[state_current_i + (action,)] = q_new
            elif done and state_new[0] >= env.goal_position: #reached goal
                #assign max q-value
                q_table[state_current_i + (action,)] = 0 
                '''
                if period_new:
                    print(f'steps: {steps}')
                '''
                
            state_current_i = state_new_i
            steps += 1
            total_rewards += reward

        progress.update(epoch, total_rewards)

        if epoch <= EPOCH_END_DECAY:
            epsilon -= EPSILON_DECAY
        
    #print('')
    env.close()
    progress.show()

if __name__ == '__main__':
    main()
