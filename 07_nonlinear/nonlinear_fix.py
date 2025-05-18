import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Shomate parameters (CO, CO₂, C(s))
A1, B1, C1, D1, E1 = 25.56759, 6.096130, 4.054656, -2.671301, 0.131021    # CO
A2, B2, C2, D2, E2 = 24.99735, 55.18696, -33.69137, 7.948387, -0.136638   # CO₂
A3, B3, C3, D3, E3 = 17.7289, 28.0988, -4.21434, 0.218050, -0.000281      # C(s)

# Δ parameters for 2CO → CO₂ + C
delta_A = A2 + A3 - 2*A1
delta_B = B2 + B3 - 2*B1
delta_C = C2 + C3 - 2*C1
delta_D = D2 + D3 - 2*D1
delta_E = E2 + E3 - 2*E1

# Standard values
delta_H_298 = -172459  # J/mol
delta_S_298 = -175.79  # J/(mol·K)
R = 8.314

def delta_H(T):
    term1 = delta_A * (T - 298)
    term2 = (delta_B / 1000) * (T**2 / 2 - 298**2 / 2)
    term3 = (delta_C / 1e6) * (T**3 / 3 - 298**3 / 3)
    term4 = (delta_D / 1e9) * (T**4 / 4 - 298**4 / 4)
    term5 = delta_E * 1e6 * (-1/T + 1/298)
    return delta_H_298 + term1 + term2 + term3 + term4 + term5

def delta_S(T):
    term1 = delta_A * np.log(T / 298)
    term2 = (delta_B / 1000) * (T - 298)
    term3 = (delta_C / 2e6) * (T**2 - 298**2)
    term4 = (delta_D / 3e9) * (T**3 - 298**3)
    term5 = delta_E * 1e6 * (-1/(2*T**2) + 1/(2*298**2))
    return delta_S_298 + term1 + term2 + term3 + term4 + term5

def f(T, x_CO):
    K = (1 - x_CO) / (x_CO**2)
    lnK = np.log(K)
    dH = delta_H(T)
    dS = delta_S(T)
    return lnK + dH/(R*T) - dS/R

# Newton-Raphson Solver with Iteration Logging
def newton_raphson(x_CO, T_guess=800, tol=1e-6, max_iter=100):
    T = T_guess
    iterations = []
    for n in range(max_iter):
        F = f(T, x_CO)
        dT = 1e-3
        F_plus = f(T + dT, x_CO)
        dFdT = (F_plus - F) / dT
        T_new = T - F / dFdT
        error = abs(T_new - T)
        iterations.append([n, T, F, error])
        # if error < tol:
        #     break
        T = T_new
    return pd.DataFrame(iterations, columns=["Iteration", "T", "f(T)", "Error"])

# Bisection Solver with Iteration Logging
def bisection(x_CO, T_low=600, T_high=1200, tol=1e-6, max_iter=100):
    iterations = []
    for n in range(max_iter):
        T_mid = (T_low + T_high) / 2
        F_mid = f(T_mid, x_CO)
        F_low = f(T_low, x_CO)
        error = T_high - T_low
        iterations.append([n, T_mid, F_mid, error])
        if F_mid * F_low < 0:
            T_high = T_mid
        else:
            T_low = T_mid
        # if error < tol:
        #     break
    return pd.DataFrame(iterations, columns=["Iteration", "T", "f(T)", "Error"])

# Solve for x_CO = [0.1, 0.3, 0.5] and save results
x_CO_values = [0.1, 0.3, 0.5, 0.7, 0.9]
results_dir = "07_nonlinear/results"
os.makedirs(results_dir, exist_ok=True)

iteration = 21
# Generate and save iteration logs
for x_CO in x_CO_values:
    df_nr = newton_raphson(x_CO, max_iter=iteration)
    df_bs = bisection(x_CO, max_iter=iteration)
    with pd.ExcelWriter(f"{results_dir}/xCO_{x_CO:.1f}_results.xlsx") as writer:
        df_nr.to_excel(writer, sheet_name="Newton-Raphson", index=False)
        df_bs.to_excel(writer, sheet_name="Bisection", index=False)

# Plot x_CO vs. T
# x_plot = np.linspace(0.1, 0.9, 50)
x_plot = x_CO_values
T_plot_nr = [newton_raphson(x, max_iter=20).iloc[-1]["T"] for x in x_plot]
# T_plot_bs = [bisection(x, max_iter=20).iloc[-1]["T"] for x in x_plot]

plt.figure(figsize=(8, 5))
plt.style.use('seaborn-v0_8-notebook')
plt.plot(x_plot, T_plot_nr, 'r-', marker='o', label="Newton")
# plt.plot(x_plot, T_plot_bs, 'b-', marker='s', label="Bisection")
plt.xlabel("$x_{\mathrm{CO}}$ = CO/(CO+CO2)")
plt.ylabel("Temperature (K)")
plt.title("Boudouard Reaction Equilibrium")
plt.grid(True)
plt.savefig(f"{results_dir}/xCO_vs_T.png")
plt.show()

# Plot error convergence for both methods
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-notebook')
# Newton-Raphson Error Plot
markerss = ['o', 's', 'd', '^', 'v', 'p', '*', 'h', '+', 'x']
i=0
# plt.subplot(1, 2, 1)
for x_CO in x_CO_values:
    df = pd.read_excel(f"{results_dir}/xCO_{x_CO:.1f}_results.xlsx", sheet_name="Newton-Raphson")
    plt.semilogy(df["Iteration"], df["Error"], marker=markerss[i], label=f"$x_{{\\mathrm{{CO}}}}$ = {x_CO}")
    i += 1
plt.xlabel("Iteration")
plt.ylabel("Error (log scale)")
plt.title("Newton Error Convergence")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/newton_error_convergence.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-notebook')
# Bisection Error Plot
# plt.subplot(1, 2, 2)
i=0
for x_CO in x_CO_values:
    df = pd.read_excel(f"{results_dir}/xCO_{x_CO:.1f}_results.xlsx", sheet_name="Bisection")
    plt.semilogy(df["Iteration"], df["Error"], marker=markerss[i], label=f"$x_{{\\mathrm{{CO}}}}$ = {x_CO}")
    i += 1
plt.xlabel("Iteration")
plt.ylabel("Error (log scale)")
plt.title("Bisection Error Convergence")
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()
plt.savefig(f"{results_dir}/bisection_error_convergence.png")
plt.close()



# Plot value convergence for both methods
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-notebook')
# Newton-Raphson value Plot
# plt.subplot(1, 2, 1)
i=0
for x_CO in x_CO_values:
    df = pd.read_excel(f"{results_dir}/xCO_{x_CO:.1f}_results.xlsx", sheet_name="Newton-Raphson")
    plt.plot(df["Iteration"], df["T"], marker=markerss[i], label=f"T at $x_{{\\mathrm{{CO}}}}$ = {x_CO}")
    i += 1
plt.xlabel("Iteration")
plt.ylabel("T (K)")
plt.title("Newton Temperature Convergence")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/Newton_Temperature_convergence.png")
plt.close()
# Bisection Error Plot
# plt.subplot(1, 2, 2)
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-notebook')
i=0
for x_CO in x_CO_values:
    df = pd.read_excel(f"{results_dir}/xCO_{x_CO:.1f}_results.xlsx", sheet_name="Bisection")
    plt.plot(df["Iteration"], df["T"], marker=markerss[i], label=f"T at $x_{{\\mathrm{{CO}}}}$ = {x_CO}")
    i += 1
plt.xlabel("Iteration")
plt.ylabel("T (K)")
plt.title("Bisection Temperature Convergence")
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()
plt.savefig(f"{results_dir}/Bisection_Temperature_convergence.png")
plt.close()