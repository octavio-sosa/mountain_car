import gym
import numpy as np
import q_value.q as q

def main():
    env = gym.make("MountainCar-v0")
    env.reset()

    q_agent = q.agent(LEARNING_RATE=0.1, DISCOUNT=0.95,
                      observation_space=env.observation_space,
                      num_val_per_observation=20) 
    q_agent.table = q.make_table(num_val_per_observation=20,
                                 len_observation_space=len(env.observation_space.high),
                                 num_actions=env.action_space.n,
                                 reward_range=(-1,0))
    EPOCHS = 2500
    PERIOD = 250
    epsilon = 0.5
    EPOCH_ZEROING_EPSILON = EPOCHS//4
    EPSILON_DECAY = epsilon/EPOCH_ZEROING_EPSILON

    for epoch in range(EPOCHS):
        if (epoch % PERIOD == 0) or (epoch == EPOCHS-1):
            new_period = True
            print(f'epoch: {epoch}')
        else:
            new_period = False

        q_agent.update_state_index('current', env.reset())

        done = False
        steps = 0
        while not done:
            if new_period:
                env.render()

            if np.random.random() > epsilon:
                q_agent.update_action('table')
            else:
                q_agent.update_action('random')

            new_state, q_agent.reward, done, _ = env.step(q_agent.action)
            q_agent.update_state_index('new', new_state)
            q_agent.update_table(done, new_state[0])
            q_agent.update_state_index('current', new_state)
            steps += 1

            if new_period and done:
                print(f'steps: {steps}\n')

        if epoch <= EPOCH_ZEROING_EPSILON:
            epsilon -= EPSILON_DECAY

    env.close()

if __name__ == '__main__':
    main()
    
