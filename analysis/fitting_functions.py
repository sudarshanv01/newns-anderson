import numpy as np
from scipy.optimize import curve_fit

def allow_epsd_inputs(func):
    """Decorator to allow eps_d values 
    to be passed to functions which need
    a filling value."""
    def wrapper(x, *args, **kwargs):
        input_epsd = kwargs.get('input_epsd', False)
        if input_epsd:
            # In this case, first convert the argument
            # from an eps_d value to a filling value.
            fitted_epsd_to_filling = kwargs.get('fitted_epsd_to_filling', None)
            if fitted_epsd_to_filling is not None:
                filling = function_linear(x, *fitted_epsd_to_filling)
                if kwargs.get('is_derivative', False):
                    # We are computing a derivative, so we need
                    # to use chain rule, which is the first term
                    # in the derivative is the filling function
                    # is linear with eps_d
                    chain_rule_prefac = fitted_epsd_to_filling[0]
                else:
                    chain_rule_prefac = 1.0
            else:
                raise ValueError('No fitting function for eps_d to filling.')
            return chain_rule_prefac * func(filling, *args)

        else:
            return func(x, *args)
    return wrapper


@allow_epsd_inputs
def func_a_by_r(x, a):
    return a / np.sqrt(x)

@allow_epsd_inputs
def func_a_by_r_derivative(x, a):
    return - a / x**1.5 / 2

def func_exp(x, a, b):
    return a * np.exp(b * x)

def func_exp_derivative(x, a, b):
    return a * b * np.exp(b * x)

@allow_epsd_inputs
def func_a_r_sq(x, a, b, c):
    return b - a * ( x - c)**2

@allow_epsd_inputs
def func_a_r_sq_derivative(x, a, b, c):
    return - 2 * a * (x - c)

def function_linear(x, a, b):
    return a * x + b

def function_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def get_fit_for_Vsd(x, y, **kwargs):
    """Prepare the fitting function and perform
    the fit for Vsd."""
    initial_guess = [1]
    popt, pcov = curve_fit(lambda x, a: func_a_by_r(x, a, **kwargs),
                            x, y, p0=initial_guess)
    return list(popt), list(pcov)

def get_fit_for_wd(x, y, **kwargs):
    """Get the fit and error for the width.""" 
    initial_guess = [2, 0.1, 0.6]
    popt, pcov = curve_fit(lambda x, a, b, c: func_a_r_sq(x, a, b, c, **kwargs),
                            x, y, p0=initial_guess)
    return list(popt), list(pcov)

def get_fit_for_epsd(x, y):
    """Get the fit and error for epsd vs. filling."""
    popt, pcov = curve_fit(function_linear, x, y)
    return list(popt), list(pcov)

def get_fitted_function(quantity_y: str):
    """For a given quantity varied against another, return
    the chosen fitting function and its derivative.""" 
    # NOTE: If the input quantity is eps_d, then it has to
    # be converted to filling before the corresponding 
    # fitted quantity can be determined.
    if quantity_y == 'wd':
        return func_a_r_sq, func_a_r_sq_derivative
    elif quantity_y == 'Vsd':
        return func_a_by_r, func_a_by_r_derivative
    else:
        raise ValueError(f'{quantity_y} are not valid quantities.')
