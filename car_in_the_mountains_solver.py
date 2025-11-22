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
from discretized_state_space import rectangle_discretized_state_space
from value_iteration_policy import value_iteration_policy_func
# =============================================================================
# 1. Functions
# =============================================================================
# Initialize the values
def initialization_car_in_mountains():
    """
    Initializes the parameter ranges and settings for the mountain car environment.
    
    Args:
        This function takes no arguments.

    Returns:
        p_tuple (tuple[float, float]) – The minimum and maximum position values.

        v_tuple (tuple[float, float]) – The minimum and maximum velocity values.

        u (list[int]) – The set of possible actions (accelerations).

        grid_num_p (int) – Number of grid points used to discretize the position space.

        grid_num_v (int) – Number of grid points used to discretize the velocity space.
    """
    p_tuple = (-1.2, 0.6)
    v_tuple = (-0.07,0.07)
    u = [-1, 0, +1]
    grid_num_p = 40
    grid_num_v = 40

    return p_tuple, v_tuple, u, grid_num_p, grid_num_v

def clip(value, min_value, max_value):
    """
    Clips a value to ensure it lies within a specified range.
    
    Args:
        value (float or np.ndarray) – The input value or array to be clipped.

        min_value (float) – The minimum allowed value.

        max_value (float) – The maximum allowed value.

    Returns:
        float or np.ndarray – The clipped value or array, constrained between min_value and max_value.
    """

    return np.clip(value, min_value, max_value)

# Transition function
def transition_calculation_car(current_p , current_v, u): 
    """
    Computes the next position and velocity of the car in the mountain car environment given the current state and action.
    
    Args:
        current_p (float) – Current position of the car.

        current_v (float) – Current velocity of the car.

        u (int or float) – Action applied to the car (acceleration).

    Returns:
        new_p (float) – Updated position after applying the action and environment dynamics.

        new_v (float) – Updated velocity after applying the action and environment dynamics.
    """
    
    new_v = clip(value = current_v + (0.001 * u) - (0.0025 * np.cos(3*current_p)), min_value = -0.07, max_value = 0.07)
    new_p = clip(value = current_p + new_v, min_value = -1.2, max_value = 0.6)
    return new_p, new_v

# Reward function
def reward_calculation_car(current_p, current_v, u, next_p, next_v):
    """
    Calculates the reward for the mountain car environment transition.
    
    Args:
        current_p (float) – Current position of the car.

        current_v (float) – Current velocity of the car.

        u (int or float) – Action applied to the car.

        next_p (float) – Next position after applying the action.

        next_v (float) – Next velocity after applying the action.

    Returns:
        int – Reward for the transition (always -1 in this implementation).
    """
    return -1

# Main function of process
def car_in_mointain_solver_func():
    """
    Solves the mountain car problem using discretization and value iteration to compute the optimal value function and policy.
    
    Args:
        This function takes no arguments.

    Returns:
        V (np.ndarray) – Optimal value function for all discretized states.

        policy (np.ndarray) – Optimal action to take at each discretized state.
    """

    # 1. Create the problem
    p_tuple, v_tuple, u, grid_num_p, grid_num_v = initialization_car_in_mountains()

    # 2. Discretize the space
    p_bins_array, v_bins_array, all_states = rectangle_discretized_state_space(p_tuple, v_tuple, grid_num_p ,grid_num_v)

    # 3. Find the optimal policy by value iteration
    V, policy = value_iteration_policy_func(transition_calculation_car, reward_calculation_car, p_bins_array, v_bins_array, all_states, u, gamma=0.95, epsilon=1e-6, max_iterations=1000)

    # 4. Return the result
    return V, policy

# =============================================================================
# Run the main
# =============================================================================     

if __name__ == '__main__':    
    print("=============== Solving Car Mountain Problem ===============") 
    V_optimal, policy_optimal = car_in_mointain_solver_func()

    print(f"Value of the best policy : {V_optimal}")
    print(f"The optimal policy : {policy_optimal}")