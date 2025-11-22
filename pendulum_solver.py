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
from discretized_state_space import rectangle_discretized_state_space
from value_iteration_policy import value_iteration_policy_func
# =============================================================================
# 1. Functions
# =============================================================================
# Initialize the values
def initialization_pendulum():
    """
    Initializes the parameter ranges and settings for the pendulum environment.
    
    Args:
        This function takes no arguments.

    Returns:
        p_tuple (tuple[float, float]) – The minimum and maximum angular positions (radians).

        v_tuple (tuple[float, float]) – The minimum and maximum angular velocities.

        u (list[int]) – The set of possible torques (actions).

        grid_num_p (int) – Number of grid points used to discretize the position space.

        grid_num_v (int) – Number of grid points used to discretize the velocity space.
    """

    p_tuple = (-np.pi, np.pi)
    v_tuple = (-10,10)
    u = [-5, 0 ,5]
    grid_num_p = 40
    grid_num_v = 40

    return p_tuple, v_tuple, u, grid_num_p, grid_num_v

# Calculate the accelration
def acceleration_calculation(current_p = 0, current_v = 0, current_u = 0, g = 9.81, m = 1 ,µ = 0.01, l = 1):
    """
    Calculates the angular acceleration of a pendulum given its current state and applied torque.
    
    Args:
        current_p (float, optional) – Current angular position (radians). Default is 0.

        current_v (float, optional) – Current angular velocity. Default is 0.

        current_u (float, optional) – Applied torque. Default is 0.

        g (float, optional) – Gravitational acceleration. Default is 9.81.

        m (float, optional) – Mass of the pendulum. Default is 1.

        µ (float, optional) – Friction coefficient. Default is 0.01.

        l (float, optional) – Length of the pendulum. Default is 1.

    Returns:
        float – The angular acceleration of the pendulum.
    """
    return (1/m*pow(l,2)) * (-(µ*current_v) + (m*g*l*(np.sin(current_p)))+ current_u)

# Calculate the new state (continues value)
def new_state_calculation(a, current_p , current_v, delta_t = 0.01): 
    """
    Computes the next angular position and velocity of a pendulum given its current state and acceleration.
    
    Args:
        a (float) – Angular acceleration.

        current_p (float) – Current angular position.

        current_v (float) – Current angular velocity.

        delta_t (float, optional) – Time step for the update. Default is 0.01.

    Returns:
        new_p (float) – Updated angular position after the time step.

        new_v (float) – Updated angular velocity after the time step.
    """

    new_v = current_v + (a * delta_t) 
    new_p = current_p + (new_v * delta_t) 
    return new_p, new_v

# Transition function
def transition_calculation_pendulum(current_p, current_v, u, delta_t = 0.01):
    """
    Calculates the next angular position and velocity of a pendulum given the current state and applied torque.
    
    Args:
        current_p (float) – Current angular position of the pendulum.

        current_v (float) – Current angular velocity of the pendulum.

        u (float) – Applied torque.

        delta_t (float, optional) – Time step for the state update. Default is 0.01.

    Returns:

        new_p (float) – Updated angular position after applying the dynamics.

        new_v (float) – Updated angular velocity after applying the dynamics.
    """
    # 1. Compute acceleration
    a = acceleration_calculation(current_p, current_v, u, g = 9.81, m = 1 ,µ = 0.01, l = 1)

    # 2. Compute next continuous state
    new_p, new_v = new_state_calculation(a, current_v, current_p, delta_t)

    return new_p, new_v

# Reward function
def reward_calculation_pendulum(current_p, current_v, u, next_p, next_v):
    """
    Calculates the reward for a pendulum transition based on the current angular position.
    
    Args:
        current_p (float) – Current angular position.

        current_v (float) – Current angular velocity.

        u (float) – Applied torque.

        next_p (float) – Next angular position after applying the torque.

        next_v (float) – Next angular velocity after applying the torque.

    Returns:

        float – Reward for the transition, computed as the cosine of the current angular position.
    """

    return np.cos(current_p)

# Main function of process
def pendulum_solver_func():
    """
    Solves the pendulum control problem using discretization and value iteration to compute the optimal value function and policy.
    
    Args:
        This function takes no arguments.

    Returns:

        V (np.ndarray) – Optimal value function for all discretized states.

        policy (np.ndarray) – Optimal action to take at each discretized state.
    """
    # 1. Create the problem
    p_tuple, v_tuple, u, grid_num_p, grid_num_v = initialization_pendulum()

    # 2. Discretize the space
    p_bins_array, v_bins_array, all_states = rectangle_discretized_state_space(p_tuple, v_tuple, grid_num_p ,grid_num_v)

    # 3. Find the optimal policy by value iteration
    V, policy = value_iteration_policy_func(transition_calculation_pendulum, reward_calculation_pendulum, p_bins_array, v_bins_array, all_states, u, gamma=0.95, epsilon=1e-6, max_iterations=1000)

    # 4. Return the result
    return V, policy
    
# =============================================================================
# Run the main
# =============================================================================     

if __name__ == '__main__':    
    print("=============== Solving Pendulum Problem ===============")  
    V_optimal, policy_optimal = pendulum_solver_func()

    print(f"Value of the best policy : {V_optimal}")
    print(f"The optimal policy : {policy_optimal}")
    print(len(policy_optimal))