import numpy as np
import cvxpy as cp
import pandas as pd
import dsp
from typing import Tuple

from cvxpy.transforms.suppfunc import SuppFunc

def solve_robust_prob(
        A: np.ndarray,
        b: np.ndarray,
        unmasked_A_df: pd.DataFrame,
        use_city_center_constraint: bool,
        city_center: np.ndarray,
        city_center_diag_matrix: np.ndarray,
        use_grid = None
)-> Tuple[np.ndarray, float]:
    """
    Solves the robust OLS problem, with the specified robust constraints.

    :param A: the feature matrix, with possibly masked entries. shape is (n, d)
    :param b: the response vector
    :param unmasked_A_df: the DF containing the unmasked A
    :param use_city_center_constraint: indicates whether to use robust constraint which has the distance from the city center
    :param city_center: numpy array of shape (2,) that has the coordinates of the city center
    :param city_center_diag_matrix: numpy array of shape (2, 2) that defines the distance metric around the city center
    :param use_grid: specifies whether to use the square robust uncertainty sets from the grid of the map of London
    """
    n, d = np.shape(A)
    assert np.shape(b) == (n,)

    assert np.shape(city_center) == (2,)
    assert np.shape(city_center_diag_matrix) == (2, 2)
    
    x = cp.Variable(d)
    intercept = cp.Variable(1)
    p = cp.Variable(n)
    constraints = [p >= 0]
    for i in range(n):
        a = A[i]
        if not np.any(np.isnan(a)):
            constraints.append(cp.abs(cp.sum(cp.multiply(a, x)) + intercept - b[i]) <= p[i])
        else:
            nan_indices, observed_indices = distinguish_indices(a)
            assert len(observed_indices) > 0
            assert len(nan_indices) == 2
            assert nan_indices[0] < nan_indices[1]

            y_loc1, y_loc2 = dsp.LocalVariable(d), dsp.LocalVariable(d)
            G_constraints1, G_constraints2 = [], []

            # observed entries constraint
            G_constraints1.append(y_loc1[observed_indices] == a[observed_indices])
            G_constraints2.append(y_loc2[observed_indices] == a[observed_indices])

            if use_grid is not None:
                grid_map, grid_center, lng_step, lat_step = use_grid
                lng_low, lat_low = grid_map[i]
                lng_low = lng_low * lng_step + grid_center[0]
                lat_low = lat_low * lat_step + grid_center[1]

                G_constraints1 += [y_loc1[nan_indices[0]] <= lng_low + lng_step, y_loc1[nan_indices[0]] >= lng_low]
                G_constraints1 += [y_loc1[nan_indices[1]] <= lat_low + lat_step, y_loc1[nan_indices[1]] >= lat_low]

                G_constraints2 += [y_loc2[nan_indices[0]] <= lng_low + lng_step, y_loc2[nan_indices[0]] >= lng_low]
                G_constraints2 += [y_loc2[nan_indices[1]] <= lat_low + lat_step, y_loc2[nan_indices[1]] >= lat_low]
    
            # city center constraint constraint
            # note that we square the distance
            if use_city_center_constraint:
                distance_from_center = unmasked_A_df.iloc[i]["dist"]
                
                G_constraints1.append(
                    cp.quad_form(y_loc1[nan_indices] - city_center, city_center_diag_matrix) <= distance_from_center ** 2
                )
                G_constraints2.append(
                    cp.quad_form(y_loc2[nan_indices] - city_center, city_center_diag_matrix) <= distance_from_center ** 2
                )

            f1 = dsp.saddle_inner(x, y_loc1)
            f2 = dsp.saddle_inner(-x, y_loc2)
        
            G1 = SuppFunc(y_loc1, G_constraints1)(x)
            G2 = SuppFunc(y_loc2, G_constraints2)(-x)
        
            constraints.append(G1 + intercept - b[i] <= p[i])
            constraints.append(G2 - intercept + b[i] <= p[i])

    obj = cp.Minimize(cp.sum_squares(p) / n)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    return x.value, intercept.value[0]

def distinguish_indices(a: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the indices where the vector a has Nan entries vs observed entries, and then returns each.

    :param a: A numpy array of shape (n,).
    :return: A Tuple where the first entry is a numpy array of the indices with Nan entries, and the second
             entry is a numpy array of the indices with non-Nan entries.
    """
    nan_indices = np.nonzero(np.isnan(a))[0]
    observed_indices = np.nonzero(np.invert(np.isnan(a)))[0]
    return nan_indices, observed_indices