import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from functools import lru_cache
import os

custom_lower = -3.5
custom_upper = 3.5

class FunctionApproximator:
    def __init__(self, func, func_name="", default_lower=-1, default_upper=1):
        self.func = func
        self.func_name = func_name or func.__name__
        self.default_lower = default_lower
        self.default_upper = default_upper
        
        # Default bounds for each polynomial type
        self.poly_bounds = {
            'general': (custom_lower, custom_upper),
            'legendre': (default_lower, default_upper),
            'chebyshev_1': (default_lower, default_upper),
            'chebyshev_2': (default_lower, default_upper),
            'laguerre': (0, np.inf),      # Laguerre is orthogonal on [0, inf)
            'hermite': (-np.inf, np.inf)   # Hermite is orthogonal on (-inf, inf)
        }
        
    def get_proper_bounds(self, poly_type):
        """Return appropriate bounds for the polynomial type"""
        return self.poly_bounds.get(poly_type, (self.default_lower, self.default_upper))
    
    @staticmethod
    def x_power_i(degree, x):
        """Standard polynomial basis x^i"""
        return x**degree
    
    @staticmethod
    def legendre_poly(degree, x):
        """Legendre polynomial P_n(x)"""
        if degree == 0:
            return np.ones_like(x)
        elif degree == 1:
            return x
        else:
            Pn_previous = np.ones_like(x)
            Pn_current = x
            for i in range(1, degree):
                Pn_new = ((2 * i + 1) * x * Pn_current - i * Pn_previous) / (i + 1)
                Pn_previous = Pn_current
                Pn_current = Pn_new
            return Pn_current
    
    def legendre_shifted(self, degree, x):
        """Shifted Legendre polynomial for arbitrary interval"""
        a, b = self.default_lower, self.default_upper
        t = (2 * x - (a + b)) / (b - a)  # Map to [-1, 1]
        return self.legendre_poly(degree, t)
    
    @staticmethod
    def chebyshev_1_poly(degree, x):
        """Chebyshev polynomial of the first kind T_n(x)"""
        if degree == 0:
            return np.ones_like(x)
        elif degree == 1:
            return x
        else:
            T_previous = np.ones_like(x)
            T_current = x
            for i in range(1, degree):
                T_new = 2 * x * T_current - T_previous
                T_previous = T_current
                T_current = T_new
            return T_current
    
    def chebyshev_1_shifted(self, degree, x):
        """Shifted Chebyshev polynomial of the first kind"""
        a, b = self.default_lower, self.default_upper
        t = (2 * x - (a + b)) / (b - a)  # Map to [-1, 1]
        return self.chebyshev_1_poly(degree, t)
    
    @staticmethod
    def chebyshev_2_poly(degree, x):
        """Chebyshev polynomial of the second kind U_n(x)"""
        if degree == 0:
            return np.ones_like(x)
        elif degree == 1:
            return 2 * x
        else:
            U_previous = np.ones_like(x)
            U_current = 2 * x
            for i in range(1, degree):
                U_new = 2 * x * U_current - U_previous
                U_previous = U_current
                U_current = U_new
            return U_current
    
    def chebyshev_2_shifted(self, degree, x):
        """Shifted Chebyshev polynomial of the second kind"""
        a, b = self.default_lower, self.default_upper
        t = (2 * x - (a + b)) / (b - a)  # Map to [-1, 1]
        return self.chebyshev_2_poly(degree, t)
    
    @staticmethod
    def laguerre_poly(degree, x):
        """Laguerre polynomial L_n(x)"""
        if degree == 0:
            return np.ones_like(x)
        elif degree == 1:
            return 1 - x
        else:
            L_previous = np.ones_like(x)
            L_current = 1 - x
            for i in range(1, degree):
                L_new = ((2 * i + 1 - x) * L_current - i * L_previous) / (i + 1)
                L_previous = L_current
                L_current = L_new
            return L_current
    
    @staticmethod
    def hermite_poly(degree, x):
        """Hermite polynomial H_n(x)"""
        if degree == 0:
            return np.ones_like(x)
        elif degree == 1:
            return 2 * x
        else:
            H_previous = np.ones_like(x)
            H_current = 2 * x
            for i in range(1, degree):
                H_new = 2 * x * H_current - 2 * i * H_previous
                H_previous = H_current
                H_current = H_new
            return H_current
    
    @staticmethod
    def weight_function(poly_type, x):
        """Proper weight functions for each orthogonal polynomial type"""
        if poly_type == "chebyshev_1":
            return 1 / np.sqrt(1 - x**2 + 1e-12)
        elif poly_type == "chebyshev_2":
            return np.sqrt(1 - x**2 + 1e-12)
        elif poly_type == "laguerre":
            return np.exp(-x)
        elif poly_type == "hermite":
            return np.exp(-x**2)
        else:  # general, legendre
            return 1.0
    
    def compute_approximation(self, degree, poly_func_name, poly_type="general"):
        """Compute approximation with proper bounds handling"""
        # Get proper bounds for this polynomial type
        lower, upper = self.get_proper_bounds(poly_type)
        
        # Warn if using default bounds for Laguerre/Hermite
        if poly_type in ['laguerre', 'hermite'] and (lower, upper) != self.poly_bounds[poly_type]:
            print(f"Warning: {poly_type} polynomials are only orthogonal over {self.poly_bounds[poly_type]}")
            print(f"Using provided bounds [{lower}, {upper}] may result in dense matrices")
        
        poly_func = getattr(self, poly_func_name)
        
        # Build the matrix and right-hand side vector
        A = np.zeros((degree + 1, degree + 1))
        Y = np.zeros(degree + 1)
        
        for i in range(degree + 1):
            for j in range(degree + 1):
                # Use memoized inner product calculation
                A[i, j] = self.inner_product_left(i, j, poly_type, poly_func_name, lower, upper)
            Y[i] = self.inner_product_right(i, poly_type, poly_func_name, lower, upper)
        
        # Check matrix condition number
        cond_number = np.linalg.cond(A)
        if cond_number > 1e10:
            print(f"Warning: High condition number ({cond_number:.2e}) for {poly_func_name} matrix")
            print("This may indicate numerical instability in the solution")
        
        # Solve the system with fallback to least squares
        try:
            a = np.linalg.solve(A, Y)
        except np.linalg.LinAlgError:
            print(f"Matrix is singular for {poly_func_name} with degree {degree}. Using least squares solution.")
            a = np.linalg.lstsq(A, Y, rcond=None)[0]
        
        # Prepare results
        # x_vals = np.linspace(max(lower, -100), min(upper, 100), 500)  # Practical limits for plotting
        x_vals = np.linspace(max(lower, custom_lower), min(upper, custom_upper), 500)
        y_approx = self.approximated_function(a, x_vals, poly_func, degree)
        y_exact = self.func(x_vals)
        
        # Calculate error metrics
        mse = np.mean((y_approx - y_exact)**2)
        max_error = np.max(np.abs(y_approx - y_exact))
        
        results = {
            'coefficients': a,
            'x_vals': x_vals,
            'y_approx': y_approx,
            'y_exact': y_exact,
            'hilbert_matrix': A,
            'rhs_vector': Y,
            'mse': mse,
            'max_error': max_error,
            'poly_type': poly_type,
            'poly_func_name': poly_func_name,
            'degree': degree,
            'bounds': (lower, upper),
            'condition_number': cond_number
        }
        
        return results
    
    def approximated_function(self, a, x, poly_func, degree):
        """Evaluate the approximated function"""
        return sum(a[i] * poly_func(i, x) for i in range(degree + 1))
    
    @lru_cache(maxsize=None)
    def inner_product_left(self, i, j, poly_type, poly_func_name, lower, upper):
        """Memoized inner product calculation for basis functions"""
        poly_func = getattr(self, poly_func_name)
        
        # For infinite bounds, we need to handle specially
        if not np.isfinite(lower) or not np.isfinite(upper):
            return self.inner_product_infinite(i, j, poly_type, poly_func_name)
            
        integrand = lambda x: poly_func(i, x) * poly_func(j, x) * self.weight_function(poly_type, x)
        result, _ = quad(integrand, lower, upper)
        return result
    
    @lru_cache(maxsize=None)
    def inner_product_right(self, i, poly_type, poly_func_name, lower, upper):
        """Memoized inner product calculation for function projection"""
        poly_func = getattr(self, poly_func_name)
        
        # For infinite bounds, we need to handle specially
        if not np.isfinite(lower) or not np.isfinite(upper):
            return self.inner_product_infinite(i, None, poly_type, poly_func_name, is_right=True)
            
        integrand = lambda x: self.func(x) * poly_func(i, x) * self.weight_function(poly_type, x)
        result, _ = quad(integrand, lower, upper)
        return result
    
    def inner_product_infinite(self, i, j, poly_type, poly_func_name, is_right=False):
        """
        Handle inner products with infinite bounds using appropriate quadrature
        for the polynomial type
        """
        poly_func = getattr(self, poly_func_name)
        
        if poly_type == "laguerre":
            # Laguerre: [0, inf) with weight e^(-x)
            if is_right:
                integrand = lambda x: self.func(x) * poly_func(i, x) * np.exp(-x)
                return quad(integrand, 0, np.inf)[0]
                # return quad(integrand, 0, 100)[0]
            else:
                integrand = lambda x: poly_func(i, x) * poly_func(j, x) * np.exp(-x)
                return quad(integrand, 0, np.inf)[0]
                # return quad(integrand, 0, 100)[0]
                
        elif poly_type == "hermite":
            # Hermite: (-inf, inf) with weight e^(-x^2)
            if is_right:
                integrand = lambda x: self.func(x) * poly_func(i, x) * np.exp(-x**2)
                return quad(integrand, -np.inf, np.inf)[0]
                # return quad(integrand, -100, 100)[0]
            else:
                integrand = lambda x: poly_func(i, x) * poly_func(j, x) * np.exp(-x**2)
                return quad(integrand, -np.inf, np.inf)[0]
                # return quad(integrand, -100, 100)[0]
        
        raise ValueError(f"Infinite bounds not supported for polynomial type {poly_type}")
    
    def save_results_to_excel(self, results, folder="02_interpolation/results"):
        """Save all results to an Excel file"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filename = f"{folder}/interpolation_{results['poly_func_name']}_deg{results['degree']}.xlsx"
        
        with pd.ExcelWriter(filename) as writer:
            # Save coefficients
            pd.DataFrame(results['coefficients'], 
                       columns=['Coefficients'],
                       index=[f'a{i}' for i in range(len(results['coefficients']))]).to_excel(
                writer, sheet_name='Coefficients')
            
            # Save Hilbert matrix
            pd.DataFrame(results['hilbert_matrix']).to_excel(
                writer, sheet_name='Hilbert_Matrix')
            
            # Save RHS vector
            pd.DataFrame(results['rhs_vector'], 
                       columns=['RHS']).to_excel(
                writer, sheet_name='RHS_Vector')
            
            # Save approximation data
            pd.DataFrame({
                'x': results['x_vals'],
                'y_exact': results['y_exact'],
                'y_approx': results['y_approx'],
                'error': results['y_approx'] - results['y_exact']
            }).to_excel(writer, sheet_name='Approximation_Data')
            
            # Save error metrics
            pd.DataFrame({
                'Metric': ['MSE', 'Max Error', 'Condition Number'],
                'Value': [results['mse'], results['max_error'], results['condition_number']]
            }).to_excel(writer, sheet_name='Error_Metrics')
    
    
    def save_method_plots_by_degree(self, results_by_method, folder="02_interpolation/plots"):
        """
        Save separate plots for each method, showing approximations at different degrees.
        results_by_method: dictionary mapping method names to list of results at various degrees
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        for method_name, method_results in results_by_method.items():
            if not method_results:
                continue  # Skip empty methods

            plt.figure(figsize=(12, 6))
            plt.style.use('seaborn-v0_8-notebook')
            x_vals = method_results[0]['x_vals']
            y_exact = method_results[0]['y_exact']
            x_val = np.linspace(min(method_results[0]['x_vals']), max(method_results[0]['x_vals']), 40)
            y_val = test_func(x_val)
            # plt.plot(x_vals, y_exact, 'k-', linewidth=2, label='Original Function')
            plt.scatter(x_val, y_val, facecolors='none', edgecolors='black', linewidths=1, label='Original Function')
            for result in method_results:
                degree = result['degree']
                y_approx = result['y_approx']
                # y_error = result['y_approx'] - result['y_exact']
                plt.plot(x_vals, y_approx, '--', linewidth=1.5, label=f"{method_name} (deg {degree})")

            plt.title(f"Interpolation using {method_name.replace('_shifted', '')} Polynomials")
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)
            filename = f"{folder}/interpolation_method_{method_name.replace(' ', '_')}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.style.use('seaborn-v0_8-notebook')
            x_vals = method_results[0]['x_vals']
            
            for result in method_results:
                degree = result['degree']
                y_error = result['y_approx'] - result['y_exact']
                plt.plot(x_vals, y_error, '--', linewidth=1.5, label=f"{method_name} (deg {degree})")

            plt.title(f"Interpolation error using {method_name.replace('_shifted', '')} Polynomials")
            plt.xlabel('x')
            plt.ylabel('error')
            plt.legend()
            plt.grid(True)
            filename = f"{folder}/interpolation_error_method_{method_name.replace(' ', '_')}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    def test_func(x):
        # return np.sin(6*x) + np.sign(np.sin(x + np.exp(2*x)))
        # return 20*np.sin(20*x)
        # return np.exp(-x**2) * np.sin(4*x)  # Compatible with Hermite
        # return np.exp(-x) * np.cos(x)     # Compatible with Laguerre
        return (1/np.sqrt(2*np.pi))*np.exp((-x**2)/2)
        # return np.exp(-x**2) * np.sin(4*x)  # Works with Hermite
        # return np.sin(6*x)+np.sign(np.sin(x+np.exp(2*x)))

    approx = FunctionApproximator(test_func, "[1/sqrt(2*phi)] * exp[(-x^2)/2]")
    
    methods = [
        ('x_power_i', 'general'),
        ('legendre_shifted', 'legendre'),
        ('chebyshev_1_shifted', 'chebyshev_1'),
        ('chebyshev_2_shifted', 'chebyshev_2'),
        ('laguerre_poly', 'laguerre'),
        ('hermite_poly', 'hermite'), 
    ]
    
    degrees = [3, 6, 9]  # Try different degrees
    
    # Dictionary to hold results grouped by method
    results_by_method = {func_name: [] for func_name, _ in methods}

    for degree in degrees:
        print(f"\n=== Computing for Degree {degree} ===")
        for func_name, poly_type in methods:
            try:
                print(f"Computing {func_name} (degree {degree})...")
                results = approx.compute_approximation(degree, func_name, poly_type)
                results_by_method[func_name].append(results)
                approx.save_results_to_excel(results)
            except Exception as e:
                print(f"Error computing {func_name} (degree {degree}): {str(e)}")

    # Now plot results by method
    approx.save_method_plots_by_degree(results_by_method)