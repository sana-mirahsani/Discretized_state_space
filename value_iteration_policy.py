# This file find the optimal policy by value iteration method
# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from discretized_state_space import find_cell, find_p_v
# =============================================================================
# 1. Value iteration function
# =============================================================================
def value_iteration_policy_func(transition_func, reward_func, p_bins, v_bins, grid_dict, actions, gamma=0.95, epsilon=1e-6, max_iterations=1000):
    """
    Value Iteration for the discretized inverted pendulum (deterministic transitions).
    """
    print("start to find the optimal policy...wait!")
    num_states = len(grid_dict)
    num_actions = len(actions)
    V = np.zeros(num_states)  # value function

    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(num_states)

        for cell_id in range(num_states):

            Q_values = []

            # 1. Get current continuous state (center of cell)
            current_p, current_v = find_p_v(cell_id, grid_dict)

            for u in actions:

                # 2. Calculate the next state
                next_p, next_v = transition_func(current_p, current_v, u)
                
                # 3. Map to cell_id
                next_cell_id = find_cell(next_p, next_v, p_bins, v_bins)
                
                # 4. Calculate reward 
                r = reward_func(current_p, current_v, u, next_p, next_v)
                
                # 5. Bellman update for deterministic transition
                Q = r + gamma * V[next_cell_id]
                Q_values.append(Q)

            # Update value of state
            V_new[cell_id] = np.max(Q_values)
            delta = max(delta, abs(V_new[cell_id] - V[cell_id]))

        V = V_new.copy()
        if delta < epsilon:
            print(f"Converged after {iteration+1} iterations.")
            break

    # Derive the policy
    policy = np.zeros(num_states, dtype=int)
    for cell_id in range(num_states):
        
        current_p, current_v = find_p_v(cell_id, grid_dict)

        Q_values = []
        for u in actions:
            
            # calculate the next state
            next_p, next_v = transition_func(current_p, current_v, u)
            # 3. Map to cell_id
            next_cell_id = find_cell(next_p, next_v, p_bins, v_bins)

            r = reward_func(current_p, current_v, u, next_p, next_v)

            Q = r + gamma * V[next_cell_id]
            Q_values.append(Q)
        policy[cell_id] = np.argmax(Q_values)

    return V, policy