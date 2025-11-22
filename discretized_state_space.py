# This file contains a function to discretizing the state space
# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
# =============================================================================
# 1. Discretized state space
# =============================================================================
def rectangle_discretized_state_space(p, v, num_grid_p, num_grid_v):
    """
    Discretizing the state space into a rectangle.

    Args:
        p : current postion
        v : current velocity
        num_grid_p : number of cells (horizontal)
        num_grid_v : number of cells (vertical)

    Returns:
        p_bins : Bins of postions
        v_bins : Bins of velocity
        grid_dict : A dictionary of cells in the grid; 
                    keys : cell_id.
                    values : List of 4 corners tuples.
    """
    
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

    # Remove the comment below to see the grid of the continues space
    """
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
    """

    return p_bins, v_bins, grid_dict

# =============================================================================
# 2. Map state (p,v) to cell_id in grid dictionary 
# =============================================================================
def find_cell(p,v,p_bins,v_bins):
    """
    Get (p,v), find the cell_id corresponding.

    Args:
        p : current postion
        v : current velocity
        p_bins : np.linspace of postion
        v_bins : np.linspace of velocity

    Returns:
        cell_id : key 
    """

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

# =============================================================================
# 3. Map state cell_id to its position and velocity
# =============================================================================
def find_p_v(cell_id, grid_dict):
    """
    Get cell_id of dictionary grid, find the p and v corresponding.

    Args:
        cell_id : Key of grid dictionary
        grid_dict : grid dictionary (discretized state space)

    Returns:
        p : position corresponding
        v : velocity corresponding 
    """

    corners = grid_dict[cell_id]
    p = (corners[0][0] + corners[1][0]) / 2
    v = (corners[0][1] + corners[3][1]) / 2

    return p, v