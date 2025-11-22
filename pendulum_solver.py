# ==========================================================
# Pendulum mdp
# state : (p,v) -> p (angular position) = (-pi, pi), v (angular velocity) = (-10, 10) 
# action : u = {-5, 0, 5} , -5 push left, 0 do nothing, 5 push right
# Goal : Keep p near to zero (upright), v near to zero (steadly)
# Reward : cos(p) 
# State space : Continuous rectangle: [-pi,pi] * [-10,10]
# How does it work:
# Every hundredth of a second, an action is taken from {-5, 0, 5}, then a (acceleration) is calculated, 
# then new p and v is calculated by new a , gives a new state (p,v)
# ==========================================================

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
def initialization_pendulum():
    p_tuple = (-np.pi, np.pi)
    v_tuple = (-10,10)
    u = [-5, 0 ,5]
    grid_num_p = 40
    grid_num_v = 40

    return p_tuple, v_tuple, u, grid_num_p, grid_num_v

# Calculate the accelration
def acceleration_calculation(current_p = 0, current_v = 0, current_u = 0, g = 9.81, m = 1 ,µ = 0.01, l = 1):

    return (1/m*pow(l,2)) * (-(µ*current_v) + (m*g*l*(np.sin(current_p)))+ current_u)

# Calculate the enwt state (continues value)
def new_state_calculation(a, current_p , current_v, delta_t = 0.01): 
    new_v = current_v + (a * delta_t) 
    new_p = current_p + (new_v * delta_t) 
    return new_p, new_v

# Transition function
def transition_calculation(current_p, current_v, u, delta_t = 0.01):
    
    # 1. Compute acceleration
    a = acceleration_calculation(current_p, current_v, u, g = 9.81, m = 1 ,µ = 0.01, l = 1)

    # 2. Compute next continuous state
    new_p, new_v = new_state_calculation(a, current_v, current_p, delta_t)

    return new_p, new_v

# Reward function
def reward_calculation(current_p):
    return np.cos(current_p)

# Main function of process
def pendulum_solver_func():
    # 1. Create the problem
    p_tuple, v_tuple, u, grid_num_p, grid_num_v = initialization_pendulum()

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
    pendulum_solver_func()