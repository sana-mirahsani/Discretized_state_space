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

# =============================================================================
# 1. Discretized state space
# =============================================================================
def rectangle_discretized_state_space(p, v, num_grid_p, num_grid_v):
    
    # Discretization parameters
    p_min, p_max = p[0], p[1]
    v_min, v_max = v[0], v[1]
    
    # Define bins
    p_bins = np.linspace(p_min, p_max, num_grid_p + 1)
    v_bins = np.linspace(v_min, v_max, num_grid_v + 1)
    
    # Create dictionary of grid cells
    grid_dict = {} # key = cell_id , value = list of corners
    cell_id = 0
    for i in range(num_grid_p):
        for j in range(num_grid_v):
            corners = [
                (p_bins[i],   v_bins[j]),     # bottom-left
                (p_bins[i+1], v_bins[j]),     # bottom-right
                (p_bins[i+1], v_bins[j+1]),   # top-right
                (p_bins[i],   v_bins[j+1])    # top-left
            ]
            grid_dict[cell_id] = corners
            cell_id += 1

    # Plot the grid
    plt.figure(figsize=(8, 6))
    for t in p_bins:
        plt.axvline(t, color='lightgray', linestyle='--', linewidth=0.8)
    for w in v_bins:
        plt.axhline(w, color='lightgray', linestyle='--', linewidth=0.8)

    plt.xlabel("Angle θ (radians)")
    plt.ylabel("Angular velocity ω (rad/s)")
    plt.title(f"Discretized State Space ({num_grid_p}x{num_grid_v} grid)")
    plt.xlim(p_min, p_max)
    plt.ylim(v_min, v_max)
    plt.grid(False)
    plt.show()

    return p_bins, v_bins, grid_dict

def acceleration_calculation(current_p = 0, current_v = 0, current_u = 0, g = 9.81, m = 1 ,µ = 0.01, l = 1):

    return (1/m*pow(l,2)) * (-(µ*current_v) + (m*g*l*(np.sin(current_p)))+ current_u)

def new_state_calculation(a, current_v, current_p , delta_t = 0.01): 
    new_v = current_v + (a * delta_t) 
    new_p = current_p + (new_v * delta_t) 
    return new_p, new_v

def initialization_pendulum():
    p_tuple = (-np.pi, np.pi)
    v_tuple = (-10,10)
    u = [-5, 0 ,5]
    grid_num_p = 10
    grid_num_v = 10

    return p_tuple, v_tuple, u, grid_num_p, grid_num_v

def transition_calculation(current_cell_id, action, p_bins, v_bins, grid_dict, delta_t = 0.01):
    # 1. Get current continuous state (center of cell)
    corners = grid_dict[current_cell_id]
    current_p = (corners[0][0] + corners[1][0]) / 2
    current_v = (corners[0][1] + corners[3][1]) / 2

    # 2. Compute acceleration
    a = acceleration_calculation(current_p, current_v, g = 9.81, m = 1 ,µ = 0.01, l = 1)

    # 3. Compute next continuous state
    new_p, new_v = new_state_calculation(a, current_v, action, current_p, delta_t)

    # 4. Map to discrete cell
    new_cell_id = find_cell(new_p, new_v, p_bins, v_bins)
    
    return new_cell_id 

def find_cell(p,v,p_bins,v_bins):

    # Find the bin index
    i = np.digitize(p, p_bins) - 1
    j = np.digitize(v, v_bins) - 1

    # Clip indices to stay inside the grid
    i = np.clip(i, 0, len(p_bins) - 2)
    j = np.clip(j, 0, len(v_bins) - 2)

    # Flatten 2D grid to 1D index
    num_v = len(v_bins) - 1   # number of vertical cells (velocity)
    cell_id = i * num_v + j

    return cell_id

def reward_calculation(current_p):
    return np.cos(current_p)

def value_iteration_pendulum(p_bins, v_bins, grid_dict, actions, gamma=0.95, epsilon=1e-6, max_iterations=1000):
    """
    Value Iteration for the discretized inverted pendulum (deterministic transitions).
    """
    num_states = len(grid_dict)
    num_actions = len(actions)
    V = np.zeros(num_states)  # value function

    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(num_states)

        for s in range(num_states):
            # Get the representative continuous (p, v) of this state
            corners = grid_dict[s]
            theta = (corners[0][0] + corners[1][0]) / 2
            omega = (corners[0][1] + corners[3][1]) / 2

            Q_values = []

            for u in actions:
                # Compute acceleration
                a = acceleration_calculation(current_v=omega, current_p=theta, current_u=u)
                # Compute next continuous state
                theta_next, omega_next = new_state_calculation(a, omega, theta)
                # Map to next discrete state
                next_s = find_cell(theta_next, omega_next, p_bins, v_bins)
                # Reward (cosine of angle)
                r = np.cos(theta)
                # Bellman update for deterministic transition
                Q = r + gamma * V[next_s]
                Q_values.append(Q)

            # Update value of state
            V_new[s] = np.max(Q_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new.copy()
        if delta < epsilon:
            print(f"Converged after {iteration+1} iterations.")
            break

    # Derive the policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        corners = grid_dict[s]
        theta = (corners[0][0] + corners[1][0]) / 2
        omega = (corners[0][1] + corners[3][1]) / 2
        Q_values = []
        for u in actions:
            a = acceleration_calculation(current_v=omega, current_p=theta, current_u=u)
            theta_next, omega_next = new_state_calculation(a, omega, theta)
            next_s = find_cell(theta_next, omega_next, p_bins, v_bins)
            r = np.cos(theta)
            Q = r + gamma * V[next_s]
            Q_values.append(Q)
        policy[s] = np.argmax(Q_values)

    return V, policy
# =============================================================================
# Main
# =============================================================================          
p_tuple = (-np.pi, np.pi)
v_tuple = (-10,10)
grid_num_p = 40
grid_num_v = 40
u = [-5, 0 ,5]

p_bins_array, v_bins_array, all_states = rectangle_discretized_state_space(p_tuple, v_tuple, grid_num_p ,grid_num_v)
V, policy = value_iteration_pendulum(p_bins_array, v_bins_array, all_states, u)
#all_states = create_states(p_bins_array,v_bins_array)