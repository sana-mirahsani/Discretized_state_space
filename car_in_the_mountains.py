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
from pendulum_solver import rectangle_discretized_state_space
# =============================================================================
# 1. Discretized state space
# =============================================================================
p_tuple = (-1.2, 0.6)
v_tuple = (-0.07 , 0.07) 
grid_num_p = 40
grid_num_v = 40
u = [-1, 0, +1]

p_bins_array, v_bins_array, all_states = rectangle_discretized_state_space(p_tuple, v_tuple, grid_num_p ,grid_num_v)
print(p_bins_array)