from maze_env import Maze
from rl_brain import QLearningTable
import matplotlib.pyplot as plt


EPISODES = 50

def q_learning():
    
    # initialize the environment and q learning model
    env = Maze()
    q_learning_model = QLearningTable(env.action_space)
    episode_steps_list = []
    episode_list = []
    
    # loop each episode to run q learning
    for episode in range(EPISODES):
        
        # initial observation
        state = env.reset()
        step_counter = 0

        while True:

            # fresh env
            env.render()

            # choose action based on observation
            action = q_learning_model.choose_action(str(state))

            # rl take action and get next observation and reward
            state_next, reward, done = env.step(action)

            # rl learn from this transition
            q_learning_model.learn(str(state), action, reward, str(state_next))

            # move to the next observation/state
            state = state_next
            step_counter += 1

            # break the while loop when it is the end of this episode
            if done:
                print(f"Episode {episode+1}/{EPISODES} finished in {step_counter} steps.")
                episode_steps_list.append(step_counter)
                episode_list.append(episode)
                break

    
    # end of the game
    print("Game Over")
    env.destroy()

    # Plot learning progress
    plt.plot(episode_list, episode_steps_list, marker='o', label="steps for each episode")
    plt.xlabel('Episode')
    plt.ylabel('Steps to Finish')
    plt.title('Q-learning Training Progress')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    q_learning()
