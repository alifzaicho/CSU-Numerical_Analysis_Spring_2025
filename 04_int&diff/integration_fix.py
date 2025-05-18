import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def left_rectangular(f, a, b, n):
    h = (b - a)/n
    x = np.linspace(a, b, n+1)[:-1]
    return h * np.sum(f(x))

def right_rectangular(f, a, b, n):
    h = (b - a)/n
    x = np.linspace(a, b, n+1)[1:]
    return h * np.sum(f(x))

def composite_trapezoid(f, a, b, n):
    h = (b - a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/2 * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

def newton_cotes(f, a, b, n, degree=2):
    if degree not in [2, 3, 4]:
        raise ValueError
    
    num_sub = n // degree
    if num_sub < 1:
        num_sub = 1
    actual_n = num_sub * degree
    
    h = (b - a) / actual_n
    total = 0.0
    
    for i in range(num_sub):
        sub_a = a + i * degree * h
        sub_b = sub_a + degree * h
        x = np.linspace(sub_a, sub_b, degree + 1)
        y = f(x)
        
        if degree == 2:
            total += h/3 * (y[0] + 4*y[1] + y[2])
        elif degree == 3: 
            total += 3*h/8 * (y[0] + 3*y[1] + 3*y[2] + y[3])
        elif degree == 4: 
            total += 2*h/45 * (7*y[0] + 32*y[1] + 12*y[2] + 32*y[3] + 7*y[4])
    
    return total

def romberg(f, a, b, max_k):
    T = np.zeros(max_k + 1)
    h = b - a
    T[0] = (h / 2) * (f(a) + f(b))
    
    for k in range(1, max_k + 1):
        h /= 2
        x_new = a + h * np.arange(1, 2**k, 2)
        T[k] = T[k-1]/2 + h * np.sum(f(x_new))
    
    for j in range(1, max_k + 1):
        for k in range(j, max_k + 1):
            T[k] = (4**j * T[k] - T[k-1]) / (4**j - 1)
    
    return T[max_k]

def legendre_poly(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*legendre_poly(n-1,x) - (n-1)*legendre_poly(n-2,x))/n

def legendre_roots_weights(n, tol=1e-15):
    roots = np.zeros(n)
    weights = np.zeros(n)
    x0 = np.cos(np.pi * (np.arange(1, n+1) - 0.25) / (n + 0.5))
    
    for i in range(n):
        x = x0[i]
        while True:
            P, dP = legendre_poly(n, x), 0
            # Calculate derivative using recurrence
            if x != 0:
                dP = n*(legendre_poly(n-1,x) - x*legendre_poly(n,x))/(1-x**2)
            else:
                dP = n*legendre_poly(n-1,x)
            
            dx = P/dP
            x -= dx
            if abs(dx) < tol:
                break
        
        roots[i] = x
        weights[i] = 2/((1-x**2)*dP**2)
    
    return roots, weights

def gauss_legendre(f, a, b, n):
    x, w = legendre_roots_weights(n)
    x_trans = 0.5*(b-a)*x + 0.5*(b+a)
    w_trans = 0.5*(b-a)*w
    return np.sum(w_trans * f(x_trans))

def chebyshev_poly(n):
    if n == 0:
        return [1]
    elif n == 1:
        return [1, 0]
    else:
        coeff = 2*np.pad(chebyshev_poly(n-1), (0,1), 'constant')
        coeff[:-2] -= chebyshev_poly(n-2)
        return coeff

def chebyshev_roots(n):
    return np.cos(np.pi * (2*np.arange(1, n+1) - 1) / (2*n))

def chebyshev_weights(n):
    return np.full(n, np.pi/n)

def gauss_chebyshev(f, a, b, n):
    roots = chebyshev_roots(n)
    weights = chebyshev_weights(n)
    
    # Transform from [-1,1] to [a,b]
    x_trans = 0.5*(b - a)*roots + 0.5*(b + a)
    w_trans = 0.5*(b - a)*weights
    
    # Account for the weight function
    return np.sum(w_trans * f(x_trans) * np.sqrt(1 - ((2*x_trans-(b+a))/(b-a))**2))

def generate_results(f, a, b, ns):
    methods = [
        ('Left Rect', left_rectangular),
        ('Right Rect', right_rectangular),
        ('Trapezoid', composite_trapezoid),
        ('Newton-Cotes 2', lambda f, a, b, n: newton_cotes(f, a, b, n, 2)),
        ('Newton-Cotes 3', lambda f, a, b, n: newton_cotes(f, a, b, n, 3)),
        ('Newton-Cotes 4', lambda f, a, b, n: newton_cotes(f, a, b, n, 4)),
        ('Romberg', lambda f, a, b, n: romberg(f, a, b, n)),
        ('Gauss-Legendre', gauss_legendre),
        ('Gauss-Chebyshev', gauss_chebyshev)
    ]
    
    exact = exact_integral(a, b)
    
    data = []
    for n in ns:
        row = {'n': n}
        for name, method in methods:
            try:
                if name in ['Gauss-Legendre', 'Gauss-Chebyshev']:
                    row[name] = method(f, a, b, n)
                else:
                    row[name] = method(f, a, b, n)
            except Exception as e:
                row[name] = np.nan
        data.append(row)
    
    results_df = pd.DataFrame(data)
    results_df['Exact'] = exact
    
    # Create error DataFrame
    error_data = {'n': ns}
    for name, _ in methods:
        error_data[name + '_error'] = np.abs((results_df[name] - exact)/exact)
    
    error_df = pd.DataFrame(error_data)
    error_df = error_df.rename(columns={c: c.replace('_error', '') for c in error_df.columns if c != 'n'})
    
    return results_df, error_df

def save_results_to_excel(results_df, error_df, folder="04_int&diff/results"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    results_filename = f"{folder}/integration_results.xlsx"
    error_filename = f"{folder}/integration_errors.xlsx"
    
    with pd.ExcelWriter(results_filename) as writer:
        results_df.to_excel(writer, sheet_name='Integration_Results', index=False)
        error_df.to_excel(writer, sheet_name='Errors', index=False)

    print(f"Results saved to {results_filename}")
    print(f"Errors saved to {error_filename}")

def plot_results(results_df, error_df, folder="04_int&diff/plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    markerss = ['o', 's', 'd', '^', 'v', 'p', '*', 'h', 'x','+']
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-notebook')

    # Plot 1: Integration Results
    i=0
    # plt.subplot(1, 2, 1)
    for col in results_df.columns:
        if col not in ['n', 'Exact']:
            plt.plot(results_df['n'], results_df[col], linewidth=1, marker=markerss[i], label=col)
            i += 1
    plt.plot(results_df['n'], results_df['Exact'], 'k--', label='Exact')
    plt.xlabel('n')
    plt.ylabel('Enthalpy (kJ/mol)')
    plt.title('Integration Results Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{folder}/integration_results.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Integration Errors

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-notebook')
    # plt.subplot(1, 2, 2)
    i = 0
    for col in error_df.columns:
        if col != 'n':
            plt.plot(error_df['n'], error_df[col], linewidth=1, marker=markerss[i], label=col)
            i+=1
    plt.xlabel('n')
    plt.ylabel('Absolute Error')
    plt.yscale("log")
    plt.title('Integration Errors Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{folder}/integration_errors.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def original_function(x):
    # return (1/np.sqrt(2*np.pi))*np.exp((-x**2)/2)
    return 57.753 + (-10.779)*1e-3*x + (-11.51) *1e5/(x**2) + 5.328*1e-6*x**2

def exact_integral(a, b):
    def integral(x):
        return 57.753*x + (-10.779)*1e-3*(x**2)/2 + (-11.51)*1e5*(-1/x) + 5.328*1e-6*(x**3)/3
    return integral(b) - integral(a)


if __name__ == "__main__":
    a = 298
    b = 1500
    ns = np.arange(1, 21)
    
    print("\nNumerical Integration Results for ∫f(x) dx from {} to {}".format(a, b))
    results_df, error_df = generate_results(original_function, a, b, ns)
    save_results_to_excel(results_df, error_df)
    
    print("Integration Results:")
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("\nIntegration Errors:")
    print(error_df.to_string(index=False, float_format="%.2e"))
    
    plot_results(results_df, error_df)


 
    # plt.style.use('seaborn')
    # plt.style.use('bmh')
    # plt.style.use('ggplot')
    # plt.style.use('Solarize_Light2')
    # plt.style.use('_classic_test_patch')
    # plt.style.use('_mpl-gallery')
    # plt.style.use('_mpl-gallery-nogrid')
    # plt.style.use('bmh')
    # plt.style.use('classic')
    # plt.style.use('dark_background')
    # plt.style.use('fast')
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')
    # plt.style.use('grayscale')
    # plt.style.use('seaborn-v0_8')
    # plt.style.use('seaborn-v0_8-bright')
    # plt.style.use('seaborn-v0_8-colorblind')
    # plt.style.use('seaborn-v0_8-dark')
    # plt.style.use('seaborn-v0_8-dark-palette')
    # plt.style.use('seaborn-v0_8-darkgrid')
    # plt.style.use('seaborn-v0_8-deep')
    # plt.style.use('seaborn-v0_8-muted')
    # plt.style.use('seaborn-v0_8-notebook') vv
    # plt.style.use('seaborn-v0_8-paper') v
    # plt.style.use('seaborn-v0_8-pastel')
    # plt.style.use('seaborn-v0_8-poster')
    # plt.style.use('seaborn-v0_8-talk')
    # plt.style.use('seaborn-v0_8-ticks') v
    # plt.style.use('seaborn-v0_8-white')
    # plt.style.use('seaborn-v0_8-whitegrid')
    # plt.style.use('tableau-colorblind10')