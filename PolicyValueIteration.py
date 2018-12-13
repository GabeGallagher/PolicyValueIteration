import numpy as np
import time
import gym
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

###################################################################################
# Policy Iteration Specific Methods                                                #
###################################################################################

"""
Returns a policy when given an environment and value function. A policy is a
function which suggests an action for each state in the environment.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode. Because it was unclear to me, 
            FrozenLane-v0 extends the Discrete class, which in turn extends the
            core class. The Discrete class comes with the following members:
            
            - nS: number of states
            - nA: number of actions
            - P: transitions (*)
            - isd: initial state distribution (**)
        
            (*) dictionary dict of dicts of lists, where
              P[s][a] == [(probability, nextstate, reward, done), ...] 
              quick PSA (get it?):
              Because Frozen lake is a stochastic environment, it is necessary to
              get possible outcomes of each action. That is what P[s][a] is.
            (**) list or array of length nS.
    
        value (numpy.array): An array with length equal to the number of states
            in the environment
        
    Returns:
        policy (numpy.array): An array with length equal to the number of states
            in the environment. Each element should be an integer from 0 to the
            number of actions available at each state. The number corresponds to
            the direction the agent will take from that state: LEFT = 0, DOWN = 1,
            RIGHT = 2, UP = 3
"""


def get_policy(env, value):
    policy = np.zeros(env.nS)

    for state in range(policy.size):
        available_actions = []
        for action in range(env.nA):
            action_val = []
            for outcome in env.P[state][action]:
                probability, next_state, reward, is_done = outcome
                action_val.append(probability * (reward + value[next_state]))
            available_actions.append(np.sum(action_val))
        policy[state] = np.argmax(available_actions)

    return policy


"""
Returns the optimal value function of an environment given a policy. A value 
function is a assigns a value for each state in the environment.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        max_iterations (int): The maximum number of times your run the algorithm
            before you admit you screwed up.
        acceptance_threshold (float): The "good enough" number. When getting the
            optimal value, the difference between sum of the values from all 
            states in the previous iteration and the sum of the values of all
            states from the current iteration. A higher number returns a less
            accurate model but does so much faster. This threshold is
            fundamentally different than in the getting the value function in
            value iteration. Due to the fact that we already have a policy here,
            the algorithm wil converge much faster.

    Returns:
        value function (numpy.array[float]): An array with length equal to the 
            number of states in the environment. Each element is a float from 
            [0, 1). The number corresponds to each states proximity to and
            potential for hitting the goal state.
"""


def get_value_function(env, policy, acceptance_threshold):
    is_converged = False
    val_function = np.zeros(env.nS)
    count = 0

    while not is_converged:
        prev_val_function = np.copy(val_function)
        count += 1

        for state in range(policy.size):
            value = []
            action = policy[state]
            for outcome in env.P[state][action]:
                probability, next_state, reward, is_done = outcome
                value.append(probability * (reward + prev_val_function[next_state]))
            val_function[state] = np.sum(value)

        if np.sum(np.fabs(prev_val_function - val_function)) <= acceptance_threshold:
            is_converged = True
            print("Value for " + policy.array2string + " found on iteration "
                  + str(count))

    return val_function


"""
Returns the optimal policy of a FrozenLake environment

This algorithm iterative converges to the best policy when given an environment.
The basic logic is:
    1. start with a policy of going in random directions at any given state
    2. get the optimal value function of the random policy 
    3. get a new policy using the optimal value function of the current policy
    4. If the two policies match, you have the optimal policy. If not, repeat 
        step 2 and 3 using the new policy until they match
        
    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        max_iterations (int): The default is set to 1000. In my testing, the 
            policy converged in under 7 iterations
        acceptance_threshold (float): The "good enough" number. When getting the
            optimal value, the difference between sum of the values from all 
            states in the previous iteration and the sum of the values of all
            states from the current iteration. A higher number returns a less
            accurate model but does so much faster.
            
    Returns:
        policy (numpy.array): Optimal policy for the given environment
"""


def policy_iteration_algorithm(env, max_iterations, acceptance_threshold):
    policy = np.random.choice(env.nA, size=env.nS)

    for iteration in range(max_iterations):
        previous_policy_val = get_value_function(env, policy, acceptance_threshold)
        new_policy = get_policy(env, previous_policy_val)

        if np.all(policy == new_policy):
            print("Policy iteration converged at iteration: " + str(iteration + 1))
            return policy

        policy = new_policy

    return policy


"""
Evaluates policy iteration algorithm for a given environment.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        iterations (int): The max number of times the value iteration runs
        acceptance_threshold (float): See description of acceptance_threshold
            in the policy_iteration_algorithm doc
            
    Returns:
        max_score (float): The best score from the evaluation
        time (float): The time it took to run the evaluation
"""


def evaluate_policy_iteration(env, iterations, render, acceptance_threshold):
    start_time = time.time()
    optimal_policy = policy_iteration_algorithm(env, iterations, acceptance_threshold)
    score = evaluate(env, optimal_policy, render)
    end_time = time.time()
    print("Best score = %0.2f. Time taken = %4.4f seconds" %
          (np.max(score), end_time - start_time))
    return np.max(score), end_time - start_time


###################################################################################
# Value Iteration Specific Methods                                                 #
###################################################################################
"""
Returns the optimal value function of an environment when there is not policy
available. A value function is a assigns a value for each state in the environment.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        max_iterations (int): The maximum number of times your run the algorithm
            before you admit you screwed up.
        acceptance_threshold (float): The "good enough" number. When getting the
            optimal value, the difference between sum of the values from all 
            states in the previous iteration and the sum of the values of all
            states from the current iteration. A higher number returns a less
            accurate model but does so much faster. I recommend a value of 0.01.
            Higher precision requires more iterations, roughly 100 more iterations
            per point of precision, ie: at my recommended 0.01, it takes about
            200 iterations to converge, about 300 at 0.001, 400 at 0.0001, and so
            on. I didn't find a significant increase in accuracy beyond 0.01.
        
    Returns:
        value function (numpy.array): An array with length equal to the number of 
            states in the environment. Each element is a float of [0, 1). The 
            number corresponds to how likely the agent is to reach the goal if it
            moves to each state.
"""


def value_iteration(env, max_iterations, acceptance_threshold):
    new_val = np.zeros(env.nS)

    for iteration in range(max_iterations):
        val = np.copy(new_val)
        for state in range(env.nS):
            action_vals = []
            for action in range(env.nA):
                outcomes = []
                for outcome in env.P[state][action]:
                    probability, next_state, reward, is_done = outcome
                    outcomes.append(probability * (reward + val[next_state]))
                action_vals.append(np.sum(outcomes))
            new_val[state] = np.max(action_vals)

        if np.sum(np.fabs(new_val - val)) <= acceptance_threshold:
            print("Value Iteration converged at iteration: " + str(iteration + 1))
            return new_val

    print("Iterated through max iterations and did not converge")
    return new_val


"""
Evaluates value iteration algorithm for a given environment.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        iterations (int): The max number of times the value iteration runs
        acceptance_threshold (float): See description of acceptance_threshold
            in the value_iteration_algorithm doc
            
    Returns:
        max_score (float): The best score from the evaluation
        time (float): The time it took to run the evaluation
"""


def evaluate_value_iteration(env, iterations, acceptance_threshold, render):
    start_time = time.time()
    optimal_val = value_iteration(env, iterations, acceptance_threshold)
    policy = get_policy(env, optimal_val)
    policy_score = evaluate(env, policy, render)
    end_time = time.time()
    print("Best score = %0.2f. Time taken = %4.4f seconds" %
          (np.mean(policy_score), end_time - start_time))
    return np.mean(policy_score), end_time - start_time


##################################################################################
# Shared Methods                                                                 #
##################################################################################
"""
The method for scoring our results. Sums up the reward values for each state our
agent hits on its playthrough. Ideally, the path with the highest reward value
is the optimal path for reaching the goal.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        policy (numpy.array): An array with length equal to the number of states
            in the environment. Each element should be an integer from 0 to the
            number of actions available at each state. The number corresponds to
            the direction the agent will take from that state: LEFT = 0, DOWN = 1,
            RIGHT = 2, UP = 3
        render (boolean): Whether or not to show the agent playing the game
        
    Return:
        total_reward (float): The total reward for a given path. This is the
            agent's metric for measuring how optimal it's playthrough was.
"""


def get_reward(env, policy, render=False):
    total_reward = 0
    state = env.reset()
    count = 0

    while True:
        if render:
            env.render()

        state, reward, done, prob = env.step(int(policy[state]))
        total_reward += reward
        count += 1
        if done:
            break

    return total_reward


"""
Evalaute how well the algorithm did. Returns an average of the scores from each
episode.

    Args:
        env (gym.core): An openAI gym environment. Expecting
            FrozenLake-v0 in default mode.
        policy (numpy.array): An array with length equal to the number of states
            in the environment. Each element should be an integer from 0 to the
            number of actions available at each state. The number corresponds to
            the direction the agent will take from that state: LEFT = 0, DOWN = 1,
            RIGHT = 2, UP = 3
        render (boolean): Whether or not to show the agent playing the game
        n_episodes (int): The number of times to run the experiment. A higher
            takes longer, but will give a more accurate sample for the quality
            of the algorithm
            
    Returns:
        average_score (float): The average score of the evalaution. The highest
            average I saw was ~0.84
"""


def evaluate(env, policy, render=False, n_episodes=100):
    scores = []
    for episode in range(n_episodes):
        scores.append(get_reward(env, policy, render))

    return np.mean(scores)


###########################################################################
if __name__ == '__main__':
    environment = gym.make('FrozenLake-v0').env

    policy_iteration_scores = []
    policy_times = []

    value_iteration_scores = []
    value_times = []

    for i in range(100):
        s, t = evaluate_value_iteration(environment, 1000, 1e-2, False)
        value_iteration_scores.append(s)
        value_times.append(t)

        s, t = evaluate_policy_iteration(environment, 1000, 1e-2, False)
        policy_iteration_scores.append(s)
        policy_times.append(t)

    print("Mean score of policy iteration: " + str(np.mean(policy_iteration_scores)) +
          " with a max score of: " + str(np.max(policy_iteration_scores)) +
          " and a mean time of: " + str(np.mean(policy_times)))

    print("Mean score of value iteration: " + str(np.mean(value_iteration_scores)) +
          " with a max score of: " + str(np.max(value_iteration_scores)) +
          " with a mean time of: " + str(np.mean(value_times)))

# The policy iteration seems to be about 18% faster that value iteration for this
# environment
