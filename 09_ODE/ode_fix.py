import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Callable, Optional, Tuple

class ODESolver:
    def __init__(self, 
                 f: Callable[[float, float], float], 
                 y0: float, 
                 t0: float, 
                 tf: float, 
                 h: float,
                 primitive: Optional[Callable[[float], float]] = None):
        """
        Initialize the ODE solver.
        
        Parameters:
        - f: The derivative function dy/dt = f(t, y)
        - y0: Initial condition y(t0) = y0
        - t0: Initial time
        - tf: Final time
        - h: Step size
        - primitive: Optional primitive function (analytical solution) for comparison
        """
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.tf = tf
        self.h = h
        self.primitive = primitive
        
        # Create time array
        self.t = np.arange(t0, tf + h, h)
        
    def euler(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using Euler's method.
        
        Returns:
        - t: Array of time points
        - y: Array of solution values
        """
        y = np.zeros(len(self.t))
        y[0] = self.y0
        
        for i in range(1, len(self.t)):
            y[i] = y[i-1] + self.h * self.f(self.t[i-1], y[i-1])
            
        return self.t, y
    
    def runge_kutta4(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using 4th order Runge-Kutta method.
        
        Returns:
        - t: Array of time points
        - y: Array of solution values
        """
        y = np.zeros(len(self.t))
        y[0] = self.y0
        
        for i in range(1, len(self.t)):
            t_prev = self.t[i-1]
            y_prev = y[i-1]
            h = self.h
            
            k1 = self.f(t_prev, y_prev)
            k2 = self.f(t_prev + h/2, y_prev + h/2 * k1)
            k3 = self.f(t_prev + h/2, y_prev + h/2 * k2)
            k4 = self.f(t_prev + h, y_prev + h * k3)
            
            y[i] = y_prev + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            
        return self.t, y
    
    def solve_and_compare(self) -> pd.DataFrame:
        """
        Solve using all methods and compare results.
        
        Returns:
        - DataFrame with all solutions
        """
        t, y_euler = self.euler()
        t, y_rk4 = self.runge_kutta4()
        
        results = {'t': t, 'Euler': y_euler, 'Runge-Kutta4': y_rk4}
        
        return pd.DataFrame(results)
    
    def plot_solutions(self, df: pd.DataFrame, title: str = 'ODE Solution Comparison'):
        """
        Plot the solutions from the DataFrame.
        """
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-notebook')
        
        plt.plot(df['t'], df['Euler'], 'b-', marker='o' ,label='Euler', alpha=0.7)
        plt.plot(df['t'], df['Runge-Kutta4'], 'r-', marker='s', label='Runge-Kutta 4', alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# Experimental data
t_exp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
CA_exp = np.array([1.0, 0.84, 0.68, 0.53, 0.38, 0.27, 0.16, 0.09, 0.04, 0.018, 0.006, 0.0025])

# Function to define the ODE for different reaction orders
def create_f(n: float, k: float) -> Callable[[float, float], float]:
    def f(t: float, CA: float) -> float:
        return -k * CA**n
    return f

# Least squares fit function
def least_squares_fit(x, y, n):
    G = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    for j in range(n+1):
        for k in range(n+1):
            G[j, k] = np.sum(x**(j + k))
        b[j] = np.sum(y * x**j)
    try:
        beta = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(G) @ b  # Fallback to pseudo-inverse
    return beta, G, b

# Calculate the derivative using finite differences
dCA_dt = np.zeros_like(CA_exp)
for i in range(1, len(CA_exp) - 1):
    dCA_dt[i] = (CA_exp[i+1] - CA_exp[i-1]) / (t_exp[i+1] - t_exp[i-1])
dCA_dt[0] = (CA_exp[1] - CA_exp[0]) / (t_exp[1] - t_exp[0])
dCA_dt[-1] = (CA_exp[-1] - CA_exp[-2]) / (t_exp[-1] - t_exp[-2])

# Avoid log of negative or zero values
valid_indices = CA_exp > 0
ln_CA = np.log(CA_exp[valid_indices])
ln_dCA_dt = np.log(-dCA_dt[valid_indices])

# Perform least squares fit for the linear equation ln(-dCA/dt) = ln(k) + n*ln(CA)
beta, G, b = least_squares_fit(ln_CA, ln_dCA_dt, 1)
n_estimate = beta[1]
k_estimate = np.exp(beta[0])

print(f"Estimated reaction order (n): {n_estimate}")
print(f"Estimated rate constant (k): {k_estimate}")

# Create directory for least squares data if it doesn't exist
os.makedirs('09_ODE/least_square_data', exist_ok=True)

# Save Gram matrix, right-hand side vector, and least squares results to Excel
with pd.ExcelWriter('09_ODE/least_square_data/least_squares_results.xlsx') as writer:
    pd.DataFrame({'ln_CA': ln_CA, 'ln_dCA_dt': ln_dCA_dt}).to_excel(writer, sheet_name='Data', index=False)
    pd.DataFrame(G, columns=[f'G_{i}' for i in range(G.shape[1])]).to_excel(writer, sheet_name='Gram_Matrix', index=False)
    pd.DataFrame({'b': b}).to_excel(writer, sheet_name='RHS_Vector', index=False)
    pd.DataFrame({'beta': beta}).to_excel(writer, sheet_name='Results', index=False)

# Plot the log-transformed data
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-notebook')
# plt.plot(ln_CA, ln_dCA_dt, 'ko', label='Experimental Data')
plt.scatter(ln_CA, ln_dCA_dt, facecolors='none', edgecolors='black', linewidths=1, label='Experimental Data')
plt.plot(ln_CA, beta[0] + beta[1] * ln_CA, 'r-', label='Least Squares Fit')
plt.xlabel(r'$\ln(C_A)$')
plt.ylabel(r'$\ln(-dC_A/dt)$')
plt.title('Log-Transformed Rate Data')
plt.legend()
plt.grid(True)

plt.text(0.75, 0.05, f'y = {beta[0]:.2f} + {beta[1]:.2f} x',
         transform=plt.gca().transAxes,
         fontsize=10,
         horizontalalignment='left',
         verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


# Create directory for plots if it doesn't exist
os.makedirs('09_ODE/plots', exist_ok=True)
plt.savefig('09_ODE/plots/log_transformed_rate_data.png')
plt.show()

# Function to find the best reaction order and constant
def find_best_order_and_constant():
    orders = [0.5, n_estimate, 0.7, 1.1]
    # k_guess = 0.22  # Initial guess for k
    # orders = n_estimate
    k_guess = k_estimate

    for n in orders:
        f = create_f(n, k_guess)
        solver = ODESolver(f, y0=1.0, t0=0, tf=11, h=1)
        df = solver.solve_and_compare()
        df['Experimental'] = CA_exp

        if 'Euler' in df.columns:
            df['Euler Error'] = np.abs(df['Euler'] - df['Experimental'])
        if 'Runge-Kutta4' in df.columns:
            df['RK4 Error'] = np.abs(df['Runge-Kutta4'] - df['Experimental'])

        print(f"Results for reaction order {n:0,.2f}:")
        print(df[['t', 'Euler', 'Runge-Kutta4', 'Experimental', 'Euler Error', 'RK4 Error']])
        print("\n")

        # Create directory for results if it doesn't exist
        os.makedirs('09_ODE/results', exist_ok=True)
        df.to_excel(f'09_ODE/results/order_{n:0,.2f}_results.xlsx', index=False)

        # Plot results for this reaction order
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-notebook')
        plt.scatter(df['t'], df['Experimental'], facecolors='none', edgecolors='black', linewidths=1, label='Experimental')
        # plt.scatter(ln_CA, ln_dCA_dt, facecolors='none', edgecolors='black', label='Experimental Data')

        plt.plot(df['t'], df['Euler'], 'b-', marker='o',label='Euler')
        plt.plot(df['t'], df['Runge-Kutta4'], 'r-',marker='s', label='Runge-Kutta 4')
        plt.xlabel('Time (hr)')
        plt.ylabel('Concentration (millimol/liter)')
        plt.title(f'Reaction Order {n:0,.2f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'09_ODE/plots/order_{n:0,.2f}_results.png')
        plt.close()

        # Plot error results for this reaction order
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-notebook')
        plt.plot(df['t'], df['Euler Error'], 'b-', marker='o', label='Euler Error')
        plt.plot(df['t'], df['RK4 Error'], 'r-', marker='s',label='RK4 Error')
        plt.xlabel('Time (hr)')
        plt.ylabel('Absolute Error')
        plt.title(f'Error Results for Reaction Order {n:0,.2f}')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f'09_ODE/plots/order_{n:0,.2f}_errors.png')
        plt.close()

# Run the analysis
find_best_order_and_constant()