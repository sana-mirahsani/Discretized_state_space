# ==========================================================
# car in the mountains mdp
# state : (p,v) -> p (current position) = (-1.2, 0.6), v (velocity) = (-0.07 ,0.07) 
# action : u = {-1, 0, +1} 
# Goal : reach P >= 0.5 in less than 200 interactions
# Note : If the final state is not reached in at most 200 steps, the system is returned to its initial state.
# Reward : -1 for each interaction
# function clip : keep p and v in their limits
# v_next = clip(v_current + 0.001 * u - 0.0025 * cos(3.p_current))
# p_next = clip(p_current + v_next)
# Initial step (fixed) : P = {- 0.6 ; - 0.4} and V = 0
# State space : Continuous rectangle: (-1.2, 0.6) * (-0.07 ,0.07) 

# How does it work:


# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from discretized_state_space import rectangle_discretized_state_space, find_cell, find_p_v
from value_iteration_policy import value_iteration_policy_func
# =============================================================================
# 1. Functions
# =============================================================================
# Initialize the values
def initialization_car_in_mountains():
    p_tuple = (-0.6, -0.4)
    v_tuple = (0,0)
    u = [-1, 0, +1]
    grid_num_p = 40
    grid_num_v = 40

    return p_tuple, v_tuple, u, grid_num_p, grid_num_v

def clip(value, min_value, max_value):
    return np.clip(value, min_value, max_value)

# Transition function
def transition_calculation(current_p , current_v, u): 

    new_v = clip(value = current_v + (0.001 * u) - (0.0025 * np.cos(3*current_p)), min_value = -0.07, max_value = 0.07)
    new_p = clip(value = current_p + new_v, min_value = -1.2, max_value = 0.6)
    return new_p, new_v

# Reward function
def reward_calculation(current_p, current_v, u, next_p, next_v):
    if next_p >= 0.5:
        return 0
    return -1

# Main function of process
def car_in_mointain_solver_func():
    # 1. Create the problem
    p_tuple, v_tuple, u, grid_num_p, grid_num_v = initialization_car_in_mountains()

    # 2. Discretize the space
    p_bins_array, v_bins_array, all_states = rectangle_discretized_state_space(p_tuple, v_tuple, grid_num_p ,grid_num_v)

    # 3. Find the optimal policy by value iteration
    V, policy = value_iteration_policy_func(transition_calculation, reward_calculation, p_bins_array, v_bins_array, all_states, u, gamma=0.95, epsilon=1e-6, max_iterations=1000)

    # 4. Print the result
    print(f"Value of the best policy : {V}")
    print(f"The optimal policy : {policy}")

# =============================================================================
# Run the main
# =============================================================================     

if __name__ == '__main__':    
    print("=============== Solving Car Mountain Problem ===============") 
    car_in_mointain_solver_func()