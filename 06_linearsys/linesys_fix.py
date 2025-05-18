import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def make_diagonally_dominant(A, b):
    n = len(b)
    A_new = A.copy()
    b_new = b.copy()
    col_order = np.arange(n)
    
    for i in range(n):
        if A_new[i, i] == 0:
            for j in range(i + 1, n):
                if A_new[j, i] != 0:
                    A_new[[i, j]] = A_new[[j, i]]
                    b_new[[i, j]] = b_new[[j, i]]
                    break
            for k in range(i + 1, n):
                if A_new[i, k] != 0:
                    A_new[:, [i, k]] = A_new[:, [k, i]]
                    col_order[[i, k]] = col_order[[k, i]]
                    break
    
    cost = -np.abs(A_new)
    row_ind, col_ind = linear_sum_assignment(cost)
    A_new = A_new[row_ind][:, col_ind]
    b_new = b_new[row_ind]
    col_order = col_order[col_ind]
    
    return A_new, b_new, col_order

def jacobi(A, b, max_iterations):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    results = []
    results.append({'k': 0, 'x': x.copy(), 'residual': np.linalg.norm(A @ x - b)})
    
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_total = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_total) / A[i, i]
        x = x_new.copy()
        residual = np.linalg.norm(A @ x - b)
        results.append({'k': k + 1, 'x': x.copy(), 'residual': residual})
    
    return pd.DataFrame(results)

def gauss_seidel(A, b, max_iterations):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    results = []
    results.append({'k': 0, 'x': x.copy(), 'residual': np.linalg.norm(A @ x - b)})
    
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_before = np.dot(A[i, :i], x_new[:i])
            sum_after = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum_before - sum_after) / A[i, i]
        x = x_new.copy()
        residual = np.linalg.norm(A @ x - b)
        results.append({'k': k + 1, 'x': x.copy(), 'residual': residual})
    
    return pd.DataFrame(results)

def SOR(A, b, max_iterations, w):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    results = []
    results.append({'k': 0, 'x': x.copy(), 'residual': np.linalg.norm(A @ x - b)})
    
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum_before = np.dot(A[i, :i], x_new[:i])
            sum_after = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - w) * x[i] + w * (b[i] - sum_before - sum_after) / A[i, i]
        x = x_new.copy()
        residual = np.linalg.norm(A @ x - b)
        results.append({'k': k + 1, 'x': x.copy(), 'residual': residual})
    
    return pd.DataFrame(results)

def steepest_descent(A, b, max_iterations):
    x = np.zeros_like(b, dtype=np.float64)
    r = b - A @ x
    residuals = [np.linalg.norm(r)]
    results = []
    results.append({
        'k': 0,
        'x': x.copy(),
        'p': r.copy(),
        'alpha': np.nan,
        'residual': residuals[0]
    })
    
    for k in range(max_iterations):
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        residual = np.linalg.norm(r)
        residuals.append(residual)
        results.append({
            'k': k + 1,
            'x': x.copy(),
            'p': r.copy(),
            'alpha': alpha,
            'residual': residual
        })
    
    return pd.DataFrame(results)

def conjugate_gradient(A, b, max_iterations):
    x = np.zeros_like(b, dtype=np.float64)
    r = b - A @ x
    p = r.copy()
    residuals = [np.linalg.norm(r)]
    results = []
    results.append({
        'k': 0,
        'x': x.copy(),
        'p': p.copy(),
        'r': r.copy(),
        'alpha': np.nan,
        'beta': np.nan,
        'residual': residuals[0]
    })
    
    for k in range(max_iterations):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        residual = np.linalg.norm(r_new)
        residuals.append(residual)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        results.append({
            'k': k + 1,
            'x': x.copy(),
            'p': p.copy(),
            'r': r_new.copy(),
            'alpha': alpha,
            'beta': beta,
            'residual': residual
        })
        r = r_new
    
    return pd.DataFrame(results)

# Matrix and vector definitions
A = np.array([
    [-0.346, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.799, 0.000, 0.000, 0.000, 0.000, 0.000, 0.888, 0.000, 0.000],
    [-0.304, -0.466, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.635, 0.548, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [-0.349, -0.535, 0.000, 0.000, 0.000, 0.000, 0.000, 0.202, 0.365, 0.000, 0.000, 0.000, 0.000, 0.000, 0.501, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, -1.000, 0.000, 0.000, 0.000, 0.157, 0.000, 0.000, 0.000, 0.112, 0.500, 0.727],
    [0.000, 0.000, 0.000, 0.000, -1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.273],
    [0.000, 0.000, 0.000, -1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.295, 1.000, 0.349, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.651, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.119, 0.680, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.241, -1.040, -0.363, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.009, -0.009, -0.009, 0.000, 0.880, 0.000, 0.000],
    [1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.150, -0.850, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
])

b = np.array([
    1.325,
    4.347,
    0.000,
    2.037,
    0.000,
    17.904,
    4.143,
    3.704,
    0.000,
    0.000,
    0.000,
    100.013,
    0.000,
    10.000,
    1.235,
    76.640
])

variables = ['w_CuFeS2', 'w_FeS2', 'w_C_Slag', 'w_Flux', 'w_Fuel', 'w_O2',
            'w_N2', 'w_Cu2S', 'w_FeS', 'w_Fe2SiO4', 'w_Sio2', 'w_Ca2SiO4',
            'w_Al2O3', 'w_Cu2O', 'w_SO2', 'w_CO2']

# Parameters
w = 0.965
max_iterations = 10

# Preprocessing
A_new, b_new, col_order = make_diagonally_dominant(A, b)
inv_col_order = np.argsort(col_order)

# Run methods
jacob_result = jacobi(A_new, b_new, max_iterations)
gs_result = gauss_seidel(A_new, b_new, max_iterations)
sor_result = SOR(A_new, b_new, max_iterations, w)
sd_result = steepest_descent(A, b, max_iterations)
cg_result = conjugate_gradient(A, b, max_iterations)

# Process results for original variables
def process_results(df, variables, inv_col_order):
    original_x = df['x'].apply(lambda x: x[inv_col_order])
    original_df = pd.DataFrame(original_x.tolist(), columns=variables)
    original_df = original_df.add_prefix('original_')
    final_df = pd.concat([df, original_df], axis=1)
    return final_df

jacob_final = process_results(jacob_result, variables, inv_col_order)
gs_final = process_results(gs_result, variables, inv_col_order)
sor_final = process_results(sor_result, variables, inv_col_order)

# Process SD and CG results
sd_expanded = pd.concat([sd_result.drop('x', axis=1), 
                         pd.DataFrame(sd_result['x'].tolist(), columns=variables)], 
                        axis=1)
cg_expanded = pd.concat([cg_result.drop('x', axis=1), 
                         pd.DataFrame(cg_result['x'].tolist(), columns=variables)], 
                        axis=1)

# Export to Excel
output_dir = "06_linearsys/results"
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, "linear_system_results.xlsx")

with pd.ExcelWriter(excel_path) as writer:
    jacob_final.to_excel(writer, sheet_name='Jacobi', index=False)
    gs_final.to_excel(writer, sheet_name='Gauss-Seidel', index=False)
    sor_final.to_excel(writer, sheet_name='SOR', index=False)
    sd_expanded.to_excel(writer, sheet_name='Steepest-Descent', index=False)
    cg_expanded.to_excel(writer, sheet_name='Conjugate-Gradient', index=False)


error = pd.DataFrame()
error['Jacobi'] = jacob_final['residual']
error['Gauss-Seidel'] = gs_final['residual']
error['SOR'] = sor_final['residual']
error['Steepest-Descent'] = sd_expanded['residual']
error['Conjugate-Gradient'] = cg_expanded['residual']
error.index = jacob_final['k']
print("Residuals Comparison:")
print(error.to_string())


def plot_residuals_comparison(error_df, output_dir):
    markerss = ['o', 's', 'd', '^', 'v', 'p', '*', 'h', '+', 'x']
    i=0
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-notebook')
    for column in error_df.columns:
        plt.semilogy(error_df.index, error_df[column], marker=markerss[i], label=column)
        i += 1
    plt.title('Residuals Comparison of Different Methods')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "residuals_comparison.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Residuals comparison plot saved to {plot_path}")










# Call the plotting function
plot_residuals_comparison(error, output_dir)