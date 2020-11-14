import q_value.q as q
import gym

def create_agent():
    try:
        q_agent = q.agent(LEARNING_RATE = 0.1, DISCOUNT = 0.95)
        q_agent.table = q.make_table(num_val_per_observation=20,
                                    len_observation_space=2,
                                    num_actions=3, reward_range=(-1,0)) 
        print('agent successfully created.')
        print(f'table shape: {q_agent.table.shape}')
    except Exception as exception:
        print('agent failed to create.')
        print(f'exception: {exception}')

def state_index():
    env = gym.make("MountainCar-v0")
    env.reset()
    num_val_per_observation = 20
    observation_size = (env.observation_space.high - env.observation_space.low)/num_val_per_observation

    try:
        index = q.get_state_index(env.reset(), env.observation_space.low, observation_size)
        print(f'index: {index}')
    except Exception as exception:
        print(f'exception: {exception}')
    finally:
        env.close()

def run():
    create_agent()
    state_index()
    
run()
