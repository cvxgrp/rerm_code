from typing import Callable, List
import cvxpy as cp
from cvxpy.transforms.suppfunc import SuppFunc
import numpy as np
from matplotlib import pyplot as plt


def form_rerm(
    f: Callable,
    theta: cp.Variable,
    y: np.ndarray,
    theta_constraints: List[cp.Constraint],
    xs: List[cp.Variable],
    x_constraints: List[List[cp.Constraint]],
    mode: str,
):
    """
    Helper function to form the robust empirical risk minimization problem.
    WARNING: the function f that the user provides is not verified to have the
    claimed curvature properties specified in the mode argument. It is
    impossible to verify this in general, so the user must be careful to
    provide a function that has the correct curvature properties.

    Args:
        f: a convex function
        theta: a CVXPY variable
        y: a vector of length n
        theta_constraints: a list of constraints on theta
        xs: a list of n scalar CVXPY variables
        x_constraints: a list of n lists of constraints on xs
        mode: "non-increasing" or "non-decreasing-sym-abs"

    Returns:
        A cvxpy problem instance for the robust empirical risk minimization
        problem. 
    """
    n = len(xs)
    assert theta.ndim <= 1
    assert len(xs) == len(x_constraints)
    assert len(y) == n

    z = cp.Variable(n)
    obj = 0.0
    constraints = theta_constraints
    for i in range(n):
        obj += f(z[i])
        G = SuppFunc(xs[i], x_constraints[i])
        if mode == "non-increasing":
            constraints += [
                G(-theta) + y[i] <= -z[i],
            ]
        elif mode == "non-decreasing-sym-abs":
            constraints += [
                G(theta) - y[i] <= z[i],
                G(-theta) + y[i] <= z[i],
            ]
        else:
            raise NotImplementedError

    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob


def f(x: cp.Variable):
    """
    An example of a non-decreasing convex function of the absolute value of x.
    This is how a user can specify a arbitrary convex function. 
    """
    return cp.square(cp.power(cp.abs(x), 1.5))


def test():
    """
    Test the form_rerm function on a simple linear regression problem.
    """
    sigma = 5  # uncertainty in x
    n = 100
    x = np.random.randn(n)
    a = 0.3
    y = a * x + np.random.randn(n) * 0.1

    theta = cp.Variable()
    theta_constraints = [theta >= 0]
    xs = [cp.Variable() for i in range(n)]
    x_constraints = [
        [
            xs[i] >= x[i] - sigma,
            xs[i] <= x[i] + sigma,
        ]
        for i in range(n)
    ]

    prob = form_rerm(
        f, theta, y, theta_constraints, xs, x_constraints, "non-increasing"
    )

    prob.solve(solver=cp.CLARABEL)
    print(theta.value)

    plt.plot(x, y, "o")
    plt.plot(x, theta.value * x, label="robust prediction")
    plt.show()


if __name__ == "__main__":
    test()
