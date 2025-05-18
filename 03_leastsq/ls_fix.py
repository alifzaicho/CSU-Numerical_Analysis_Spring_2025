import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load merged_data.csv
df = pd.read_csv('03_leastsq/merged_data.csv')
desc = df.describe()
print (desc)
 # Transpose to have dependent variables as rows


# Ensure directories exist
os.makedirs('03_leastsq/plots', exist_ok=True)
os.makedirs('03_leastsq/results', exist_ok=True)
os.makedirs('03_leastsq/plots/bar_charts', exist_ok=True) 

desc.to_excel("03_leastsq/results/merged_data_desc.xlsx")

y_cols = ['ys_value', 'uts_value', 'elo_value', 'roa_value']
X_cols = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Ti', 'Al', 'B', 'N', 'V', 'Co', 'Nb+Ta', 'temperature']


# Drop non-numeric columns (e.g., "composition") if needed
df_numeric = df.drop(columns=['composition'], errors='ignore')

# Get all variable names (independent + dependent)
all_variables = df_numeric.columns.tolist()

# Plot settings
plt.figure(figsize=(20, 25))  # Adjust figure size
plt.suptitle("Distribution of All Variables (Box Plots)", fontsize=20, y=1.02)

# Plot box plots for all variables in a grid
n_cols = 3  # Number of columns in the grid
n_rows = (len(all_variables) + n_cols - 1) // n_cols  # Calculate rows

for idx, var in enumerate(all_variables, 1):
    plt.subplot(n_rows, n_cols, idx)
    plt.boxplot(df_numeric[var].dropna(), vert=True, patch_artist=True)
    plt.title(f"{var}", fontsize=12)
    plt.grid(True, alpha=0.3)

plt.savefig('03_leastsq/plots/bar_charts/statdesc.png')
plt.close()
# Define your manual least squares function
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

# y_cols = ['ys_value', 'uts_value', 'elo_value', 'roa_value']
# X_cols = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'Ti', 'Al', 'B', 'N', 'V', 'Co', 'Nb+Ta', 'temperature']

# Define y-axis limits for each dependent variable
ylims_dict = {
    'ys_value': (0, 400),
    'uts_value': (0, 800),
    'elo_value': (0, 120),
    'roa_value': (0, 120)
}

# Initialize a dictionary to collect beta coefficients
beta_results = {}

# Try multiple degrees
degrees = [1]

# Loop over each independent variable
for x_name in X_cols:
    x = df[x_name].values

    # Loop through each dependent variable
    for y_name in y_cols:
        y = df[y_name].values

        # Create subdirectory for the current y_name under plots
        plot_subdir = os.path.join('03_leastsq/plots', y_name)
        os.makedirs(plot_subdir, exist_ok=True)  # Ensure the directory exists

        # Filter out NaNs in both x and y simultaneously
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) == 0:
            print(f"No valid data for {x_name} vs {y_name}")
            continue

        # Try each degree
        for degree in degrees:
            print(f"Processing {x_name} vs {y_name}, Degree={degree}")

            if len(x_valid) <= degree:
                print(f"Not enough data ({len(x_valid)}) for degree {degree}")
                continue

            try:
                beta, G, b = least_squares_fit(x_valid, y_valid, degree)
            except np.linalg.LinAlgError:
                print(f"Singular matrix for {x_name} vs {y_name}, skipping")
                continue

            # Collect beta coefficients
            if y_name not in beta_results:
                beta_results[y_name] = {}
            beta_results[y_name][x_name] = {
                'beta_0': beta[0] if len(beta) > 0 else None,
                'beta_1': beta[1] if len(beta) > 1 else None
            }

            # Generate prediction curve
            x_fit = np.linspace(min(x_valid), max(x_valid), 200)
            y_fit = np.polyval(beta[::-1], x_fit)

            # Compute residuals and L2 norm
            y_pred = np.polyval(beta[::-1], x_valid)
            residuals = y_valid - y_pred
            l2_norm = np.linalg.norm(residuals)

            # Plot actual vs predicted
            plt.figure(figsize=(8, 5))
            plt.style.use('seaborn-v0_8-notebook')
            plt.scatter(x_valid, y_valid, facecolors='none', edgecolors='black', linewidths=1, label='Data', alpha=0.6)
            plt.plot(x_fit, y_fit, 'r-', label=f'Fit (deg={degree})')
            plt.title(f"{x_name} vs {y_name} (Degree {degree})\n{beta[0]:.2f} + {beta[1]:.2f}x \n L2 Norm : {l2_norm:.2f}")
            plt.xlabel(x_name)
            plt.ylabel(y_name)

            # Set y-axis limits based on the dependent variable
            plt_ylim = ylims_dict.get(y_name, (min(y_valid), max(y_valid)))
            plt.ylim(plt_ylim)

            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save plot into the y_name subdirectory
            filename_plot = f"{x_name}_vs_{y_name}_deg{degree}.png"
            plt.savefig(os.path.join(plot_subdir, filename_plot))
            plt.close()

            # Save results to Excel
            folder = "03_leastsq/results/"
            filename_excel = f"{folder}/{x_name}_vs_{y_name}_deg{degree}.xlsx"

            with pd.ExcelWriter(filename_excel) as writer:
                # Save coefficients
                pd.DataFrame(beta,
                             columns=['Coefficients'],
                             index=[f'a{i}' for i in range(len(beta))]).to_excel(
                    writer, sheet_name='Coefficients')

                # Save Gram Matrix
                pd.DataFrame(G).to_excel(writer, sheet_name='Gram_Matrix')

                # Save RHS vector
                pd.DataFrame(b, columns=['RHS']).to_excel(writer, sheet_name='RHS')

                # Save L2 norm
                pd.DataFrame({'L2_Norm': [l2_norm]}).to_excel(writer, sheet_name='Metrics', index=False)

# After processing all variables, save beta_results to Excel
beta_df = pd.DataFrame(beta_results).T  # Transpose to have dependent variables as rows
beta_df.to_excel("03_leastsq/results/beta_coefficients.xlsx")

print("Beta coefficients collected and saved.")

# Generate bar charts for each dependent variable
for y_name in y_cols:
    if y_name not in beta_results:
        print(f"No beta results for {y_name}")
        continue

    # Prepare data for bar chart
    x_vars = []
    beta1_values = []
    for x_name in X_cols:
        if x_name in beta_results[y_name]:
            x_vars.append(x_name)
            beta1 = beta_results[y_name][x_name]['beta_1']
            beta1_values.append(beta1 if beta1 is not None else 0)  # Replace None with 0 for plotting

    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-notebook')
    bars = plt.bar(x_vars, beta1_values, color='blue')

    # Add labels and title
    plt.xlabel('Independent Variables')
    plt.ylabel('Beta_1 Coefficient')
    plt.yscale('symlog')
    plt.title(f'Beta_1 Coefficients for {y_name} by Independent Variables')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the bar chart
    bar_chart_path = os.path.join('03_leastsq/plots/bar_charts', f"{y_name}_beta1_bar_chart.png")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

print("Bar charts for beta_1 coefficients generated.")