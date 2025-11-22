# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from discretized_state_space import rectangle_discretized_state_space, find_cell
from value_iteration_policy import value_iteration_policy_func
from pendulum_solver import initialization_pendulum, reward_calculation_pendulum, transition_calculation_pendulum
from car_in_the_mountains_solver import initialization_car_in_mountains, transition_calculation_car, reward_calculation_car

# =============================================================================
# 1. Create the trajectory
# =============================================================================
def trajectory(transition_calculation, reward_calculation, policy_optimal, u, P0, V0, p_bins, v_bins, gamma, max_steps, termination_condition=None):
    
    G = 0
    traj = []
    gamma_t = 1 
    current_p, current_v = P0, V0

    for _ in range(max_steps):

        if termination_condition is not None:
            if termination_condition(current_p, current_v):
                break

        current_cell_id = find_cell(current_p, current_v, p_bins, v_bins)
        action_index = policy_optimal[current_cell_id]
        current_action = u[action_index]
        next_p, next_v = transition_calculation(current_p, current_v, current_action)

        r = reward_calculation(current_p, current_v, current_action, next_p, next_v)
        G += gamma_t * r

        next_cell_id = find_cell(next_p, next_v, p_bins, v_bins)
        traj.append((current_cell_id, current_action, r, next_cell_id))

        current_p, current_v = next_p, next_v 

        gamma_t *= gamma

    return traj, G

# =============================================================================
# 2. 20 run episode
# =============================================================================
def twenty_run_episode(transition_calculation_func, reward_calculation_func, initialization_func, first_p_func, first_v_func, gamma, max_step, termination_condition=None):
    # 1. Initialize the states, actions, space
    p_tuple, v_tuple, u, grid_num_p, grid_num_v = initialization_func()

    # 2. Discretize the space
    p_bins_array, v_bins_array, all_states = rectangle_discretized_state_space(p_tuple, v_tuple, grid_num_p ,grid_num_v)

    # 3. Find the optimal policy by value iteration
    V_optimal, policy_optimal = value_iteration_policy_func(transition_calculation_func, reward_calculation_func, p_bins_array, v_bins_array, all_states, u, gamma=0.95, epsilon=1e-6, max_iterations=1000)

    # 4. Run 20 times
    list_of_R = []
    for _ in range(20):
        P0, V0 =  first_p_func(), first_v_func()
        traj, R = trajectory(transition_calculation_func, reward_calculation_func, policy_optimal, u, P0, V0, p_bins_array, v_bins_array, gamma, max_step, termination_condition)
        list_of_R.append(R)

    return list_of_R

# =============================================================================
# 3. Plot twenty values of R and find the median
# =============================================================================
def plot_returns(R_list, title="Distribution of Returns"):
    """
    Plot the distribution of return values and return the median.

    Args:
        R_list (list or np.array): List of return values
        title (str): Plot title

    Returns:
        float: Median of the return values
    """
    R_array = np.array(R_list)
    median_R = np.median(R_array)
    
    plt.figure(figsize=(8,5))
    plt.hist(R_array, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(median_R, color='red', linestyle='--', label=f"Median = {median_R:.2f}")
    plt.title(title)
    plt.xlabel("Return R")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return median_R

# =============================================================================
# 4. Run test for pendulum problem
# =============================================================================
def pendulum_problem():
    print("=============== Twenty Run Episode for Pendulum Problem ===============")  
    def first_p():
        return np.random.uniform(-np.pi, np.pi)

    def first_v():
        return np.random.uniform(-10, 10)

    R_list = twenty_run_episode(transition_calculation_pendulum, reward_calculation_pendulum, initialization_pendulum, first_p, first_v, 0.9, 100)
    
    R_median = plot_returns(R_list, title="Distribution of Returns")
    print(f"========= R_median : {R_median} =========")

# =============================================================================
# 5. Run test for car in mountain problem
# =============================================================================
def car_in_mountain_problem():
    print("=============== Twenty Run Episode for Car In Mountain Problem ===============")  
    def first_p():
        return np.random.uniform(-0.6, -0.4)

    def first_v():
        return 0

    def termination_condition(current_p, current_v):
        return current_p >= 0.5
    R_list = twenty_run_episode(transition_calculation_car, reward_calculation_car, initialization_car_in_mountains, first_p, first_v, 0.9, 200, termination_condition)
    print(R_list)
    R_median = plot_returns(R_list, title="Distribution of Returns")
    print(f"========= R_median : {R_median} =========")

# =============================================================================
# Run the main
# =============================================================================     
if __name__ == '__main__':    
    #pendulum_problem()

    # Remove the comment to run the car_in_mountain function
    car_in_mountain_problem()
