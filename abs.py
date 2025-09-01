from scipy.optimize import linprog

# Define the coefficients of the objective function
# Example: Minimize c1*x1 + c2*x2
c = [1, 2]

# Define the inequality constraints (A_ub * x <= b_ub)
# Example: 2*x1 + x2 <= 20, 4*x1 + 3*x2 <= 30
A_ub = [
    [2, 1],
    [4, 3]
]
b_ub = [20, 30]

# Define the bounds for each variable
# Example: 0 <= x1 <= 10, 0 <= x2 <= 10
x_bounds = [(0, 10), (0, 10)]

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

# Check if the optimization was successful
