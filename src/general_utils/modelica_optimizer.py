from .modelica_integrator import modelica_integrator
from scipy.optimize import differential_evolution, minimize
import numpy as np
import psutil
import pandas as pd
import logging

def exponential_penalty(x, lower_bound, upper_bound, growth_rate = 2):
    # Calculate the exponential penalty based on the violation
    violation_lower = max(0, (lower_bound - x)/lower_bound) if lower_bound != -np.inf else max(0, lower_bound - x)
    violation_upper = max(0, (x - upper_bound)/upper_bound) if upper_bound != np.inf else max(0, x - upper_bound)
    
    penalty_lower = np.exp(growth_rate * violation_lower)-1
    penalty_upper = np.exp(growth_rate * violation_upper)-1

    objective_contribution = penalty_lower + penalty_upper
    
    return objective_contribution

def process_array(input_array, substitution_array):
    # Check for NaN values in the input array
    nan_mask = np.isnan(input_array)
    # Substitute NaN values with corresponding values from the substitution array
    input_array[nan_mask] = substitution_array[nan_mask]
    return input_array

def is_string_in_file(file_path, target_string):
    with open(file_path, 'r') as file:
        for line in file:
            if target_string in line:
                return True
    return False

def min_error(y_df: pd.DataFrame, target_values: pd.DataFrame, **kwargs):
        '''
        Compute mean squared error per output between the simulated `y_df` and
        a `target_values` DataFrame. Only columns present in both DataFrames are
        evaluated. The two DataFrames must have the same length (number of rows).

        Parameters:
        - y_df: pandas DataFrame with simulated outputs (columns are variable names).
        - target_values: pandas DataFrame with target time-series for outputs. Must
            have the same number of rows as `y_df`.
        - kwargs: optional, not used here except for compatibility; e.g. callers
            may still pass 'outputs_extract_names' in kwargs but it's not required.

        Returns:
        - cost_dict: dict mapping each common column name -> mean squared error.
        '''
        if not isinstance(target_values, pd.DataFrame):
                raise TypeError('target_values must be a pandas DataFrame')

        if len(y_df) != len(target_values):
                raise ValueError('y_df and target_values must have the same number of rows')

        common_cols = [c for c in y_df.columns if c in target_values.columns]
        if not common_cols:
                raise ValueError('No matching columns between y_df and target_values')
        # Extract log from kwargs, default to False
        log = kwargs.get('log', True)

        cost_dict = {}
        for col in common_cols:
            # If col type is date/time, skip
            if not np.issubdtype(y_df[col].dtype, np.number):
                if log:
                        logging.info(f"Skipping non-numeric column '{col}' in cost computation.")
                continue
            sim = y_df[col].to_numpy()
            tgt = target_values[col].to_numpy()
            cost_dict[col] = np.mean(((sim - tgt)/tgt) ** 2)
        return cost_dict

def enforce_constraints(y_df: pd.DataFrame, constraints: list = None,
                        weight_default: float = 50, **kwargs) -> dict:
    """
    Evaluate constraint violations on y_df and return a dict of penalty entries.

    Supports constraint dicts with types 'col' | 'comb' | 'custom'. Uses
    exponential_penalty and supports either separate 'lower'/'upper' or a box
    'bounds' = [lower, upper].
    """
    penalty_dict = {}
    if not constraints:
        return penalty_dict

    log = kwargs.get('log', True)

    # helper to reduce array to scalar
    def _reduce(vec, agg: str):
        if agg == 'max':
            return float(np.max(vec))
        if agg == 'sum':
            return float(np.sum(vec))
        if agg == 'mean':
            return float(np.mean(vec))
        raise ValueError(f"Unknown agg method '{agg}'")

    for c in constraints:
        ctype = c.get('type', 'col')
        name = c.get('name', f"constraint_{np.random.randint(1e9)}")
        weight = c.get('weight', weight_default)
        agg = c.get('agg', 'max')

        # compute the series to check
        if ctype == 'col':
            col = c['col']
            if col not in y_df.columns:
                penalty_dict[f'penalty_{name}'] = float('inf')
                continue
            series = y_df[col].to_numpy()

        elif ctype == 'comb':
            cols = c['cols']
            coeffs = c.get('coeffs', [1.0] * len(cols))
            if any(col not in y_df.columns for col in cols):
                penalty_dict[f'penalty_{name}'] = float('inf')
                continue
            arrs = [y_df[col].to_numpy() for col in cols]
            series = np.zeros_like(arrs[0], dtype=float)
            for a, coef in zip(arrs, coeffs):
                series += coef * a

        elif ctype == 'custom':
            fn = c.get('fn', None)
            if fn is None or not callable(fn):
                raise ValueError("Custom constraint requires a callable 'fn' returning per-row values or a scalar")
            result = fn(y_df)
            series = np.asarray(result)
            # If fn returns a scalar (e.g., df['col'].max()), treat as single-element array
            if series.ndim == 0:
                series = np.array([series])
            elif series.shape[0] != len(y_df):
                raise ValueError(
                    f"Custom constraint function returned array of length {series.shape[0]}, expected {len(y_df)} or a scalar"
                )
        else:
            raise ValueError(f"Unknown constraint type '{ctype}'")

        # Support box constraint syntax: 'bounds': [lower, upper]
        bounds = c.get('bounds', None)
        if bounds is not None:
            lower, upper = bounds[0], bounds[1]
        else:
            lower = c.get('lower', None)
            upper = c.get('upper', None)

        # Get growth_rate for exponential penalty (default follows current code)
        growth_rate = c.get('growth_rate', 2.0)

        # Compute exponential penalty per element
        penalty_array = np.zeros_like(series, dtype=float)
        for i, val in enumerate(series):
            lb = lower if lower is not None else -np.inf
            ub = upper if upper is not None else np.inf
            penalty_array[i] = exponential_penalty(val, lb, ub, growth_rate)
            if (val < lb or val > ub) and log:
                logging.info(f"Constraint '{name}' violated at index {i}: value {val} outside [{lb}, {ub}]\n")

        # reduce penalty to scalar and scale by weight
        scalar_penalty = _reduce(penalty_array, agg)
        penalty = weight * scalar_penalty
        penalty_dict[f'penalty_{name}'] = float(penalty)

    return penalty_dict

def min_error_constrained(y_df: pd.DataFrame, target_values: pd.DataFrame, constraints: list = None,
                          weight_default: float = 50, **kwargs) -> dict:
    """
    Compute MSE per-column between y_df and target_values, and add penalties for constraint violations.

    - y_df, target_values: pandas DataFrame (same length). Columns in common are evaluated.
    - constraints: list of constraint dicts. Each constraint dict may be one of:
        1) {'type':'col', 'col':'y1', 'lower': L, 'upper': U, 'weight': w, 'name': 'y1_bound'}
           => checks y_df['y1'] against lower/upper (per-row). Penalty uses max violation over time (or sum, see 'agg').
        2) {'type':'comb', 'cols':['y1','y2'], 'coeffs':[1,-1], 'lower': L, 'upper': U, 'weight': w, 'name': 'y1_minus_y2'}
           => checks linear combo c = sum(coeffs[i]*y_df[cols[i]]) per row.
        3) {'type':'custom', 'fn':callable, 'name':'custom1', 'weight': w}
           => 'fn' is called as fn(y_df) and must return per-row series (or a scalar). The same lower/upper semantics apply if provided in dict.
       Additional optional keys:
         - 'agg': 'max'|'sum'|'mean' (how to reduce per-row violations to a scalar). Default 'max'.
    - weight_default: default multiplier for penalties (use large to enforce).
    Returns a dict with per-output MSE keys plus 'penalty_<name>' entries when violated.

    Notes about penalty computation:
    - Penalties use an exponential growth based on the percentage degree of violation. This ensures that small violations incur small penalties,
      while larger violations lead to rapidly increasing penalties, effectively discouraging constraint breaches.
      Scale base terms (e.g. MSE) so that a good fit is between 1-100.
      Set weight to have 10-50 times larger penalties for typical violations compared to base MSE terms.
      Set growth_rate to have violation of 100% lead to exponential be around 5. Then apply weight.
      At typical violations, penalty should be noticeable but not 10⁶ times the base cost.
      growth_rate must not be crazy high (keep it ≤ 5 for most cases).
    """

    if not isinstance(target_values, pd.DataFrame):
        raise TypeError("target_values must be a pandas DataFrame")
    if len(y_df) != len(target_values):
        raise ValueError("y_df and target_values must have the same number of rows")
    # Extract log from kwargs, default to True (consistent with modelica_integrator.py)
    log = kwargs.get('log', True)

    # Base MSE cost for matching columns
    common_cols = [c for c in y_df.columns if c in target_values.columns]
    cost_dict = {}
    for col in common_cols:
        if not np.issubdtype(y_df[col].dtype, np.number):
            if log:
                logging.info(f"Skipping non-numeric column '{col}' in cost computation.")
            continue
        sim = y_df[col].to_numpy()
        tgt = target_values[col].to_numpy()
        cost_dict[col] = float(np.mean(((sim - tgt)/tgt) ** 2))

    # Evaluate constraints (if any) via reusable helper
    penalties = enforce_constraints(y_df, constraints, weight_default, **kwargs)
    if penalties:
        total_base = sum(cost_dict.values()) if cost_dict else 0.0
        total_pen = sum(penalties.values())
        if log:
            increase_pct = (total_pen / total_base * 100) if total_base > 0 else float('inf')
            logging.info(f"Overall objective function increase by {increase_pct}% from constraint penalties.")
        cost_dict.update(penalties)
    return cost_dict

class modelica_optimizer:
    """Class wrapper for optimizing Modelica model parameters.

    The optimizer constructs a param_dict for each evaluation and forwards
    it to `modelica_integrator` via `integrator_kwargs` (which must include
    all required integrator inputs except `param_dict`). The cost function is
    called with the simulation DataFrame and any cost-specific kwargs.
    """

    def __init__(self, cost_function: callable, initial_guesses: np.ndarray, param_bounds: dict,
                    cost_args: dict = None, integrator_kwargs: dict = None):
        """Initialize the optimizer.

        See module-level docstrings for parameter meanings.
        """
        self.cost_function = cost_function
        self.initial_guesses = np.asarray(initial_guesses)
        self.param_bounds = param_bounds

        # integrator kwargs (do NOT include 'param_dict' here)
        self.integrator_kwargs = integrator_kwargs or {}
        # cost-specific kwargs
        self.cost_args = cost_args or {}

        # DataFrame that will store param values and cost dictionary at each call
        self.iter_df = pd.DataFrame()
        # Attribute to store the latest simulation DataFrame
        self.y_df = pd.DataFrame()

    def cost_function_handler(self, param_values: np.ndarray):
        """Objective function for the optimizer.
        Parameters:
        - param_values is an array-like matching the order of keys in param_bounds
        when param_bounds is a dict, or the order of the provided sequence when
        param_bounds is a sequence.
        Returns:
        - A float representing the cost for the given parameter values.
        Note: if self.iter_df is empty, there will be an issue if modelica integrator fails at the very first run.
        If the outputs the user is computing the errors of are different from the one extracted from modelica_integrator and/or
        there are constraints (columns added to self.iter_df), an issue in adding rows to self.iter_df will arise as soon as the integrator will not fail anymore.
        Anyway, the user can take the last parameter values that doesn't make the integrator fail and restart again iterations!
        """
        param_values = process_array(np.asarray(param_values), self.initial_guesses)
        param_dict = {key: value for key, value in zip(self.param_bounds.keys(), param_values)} if isinstance(self.param_bounds, dict) else {i: v for i, v in enumerate(param_values)}

        # Ensure the integrator kwargs are present
        self._validate_integrator_kwargs()

        try:
            y_df, _ = modelica_integrator(param_dict=param_dict, **self.integrator_kwargs)
            # Add y_df to attribute for later access
            self.y_df = y_df
        except (Exception, ValueError, psutil.TimeoutExpired) as e:
            logging.error(f"Error occurred during Modelica integration: {e}\n"
                          "The values of the cost dictionary will be set to infinity for this evaluation: the output names in the cost function must match the ones extracted by the modelica integrator to guarantee continuity.")
            outputs_extract_names = self.integrator_kwargs.get('outputs_extract_names', []) # This can actually be different than the all ones extracted by modelica integrator
            # If self.iter_df is not empty, extract the keys from there to ensure continuity
            if not self.iter_df.empty:
                cost_dict = {name: 1e12 for name in self.iter_df.columns} # 1e12 is used instead of np.inf to avoid overflow in some optimizers
            else:
                cost_dict = {name: 1e12 for name in outputs_extract_names} # This can lead the code to stop when modelica does not fail anymore but failed at first iterations!!
            total_cost = sum(cost_dict.values())
        else:
            # modelica_integrator succeeded, now call cost_function
            try:
                # Call the cost function: pass simulation dataframe, cost args and integrator kwargs
                cost_dict = self.cost_function(self.y_df, **(self.cost_args or {}), **(self.integrator_kwargs or {}))
                
                # If constraints provided and not already applied by the base function, append penalties here
                if isinstance(cost_dict, dict):
                    has_penalties = any(str(k).startswith('penalty_') for k in cost_dict.keys())
                    constraints = None
                    if isinstance(self.cost_args, dict):
                        constraints = self.cost_args.get('constraints')
                    if constraints and not has_penalties:
                        weight_default = self.cost_args.get('weight_default', 50)
                        log_flag = self.integrator_kwargs.get('log', True)
                        penalties = enforce_constraints(self.y_df, constraints, weight_default, log=log_flag)
                        if penalties:
                            cost_dict.update(penalties)
                    total_cost = sum(cost_dict.values())
                else:
                    total_cost = cost_dict
            except Exception as e:
                raise ValueError(f"Error occurred during self.cost_function call: {e}")

        # Record evaluation in iter_df: merge params and costs into one flat dict
        record = {}
        # Extract param_scale_dict from integrator kwargs to recompute original parameter values
        param_scale_dict = self.integrator_kwargs.get('param_scale_dict', {})
        for k, v in param_dict.items():
            # If param keys are numeric (sequence bounds) leave as-is
            if isinstance(k, int):
                record[f'param_{k}'] = v / param_scale_dict.get(k)
            else:
                record[k] = v / param_scale_dict.get(k)

        # Cost columns
        if isinstance(cost_dict, dict):
            for ck, cv in cost_dict.items():
                record[ck] = cv
        else:
            record['cost'] = cost_dict

        record['total_cost'] = total_cost

        # Append to DataFrame
        self.iter_df = pd.concat([self.iter_df, pd.DataFrame([record])], ignore_index=True)

        return total_cost

    def _validate_integrator_kwargs(self):
        """Ensure that integrator_kwargs contains required keys for modelica_integrator.

        Raises a ValueError listing missing keys if any required ones are absent.
        """
        required = [
            'mo_path', 'model_name', 'folder_name', 'param_scale_dict',
            'x0_dict', 'time_interval', 'start_time', 'stop_time', 'tolerance'
        ]
        missing = [k for k in required if k not in self.integrator_kwargs]
        if missing:
            raise ValueError(f"Missing required modelica_integrator kwargs: {missing}. "
                                "Provide these (as integrator_kwargs) when constructing the optimizer; they are forwarded to the integrator.")

    def _build_bounds(self):
        """Return bounds as a list of (min, max) pairs suitable for SciPy.

        Accepts either a dict mapping parameter names -> (min, max) pairs, or a
        sequence of (min, max) pairs already in the desired order.
        """
        if isinstance(self.param_bounds, dict):
            return [self.param_bounds[key] for key in self.param_bounds.keys()]
        bounds = list(self.param_bounds)
        for b in bounds:
            if not (isinstance(b, (list, tuple)) and len(b) == 2):
                raise TypeError('param_bounds must be either a dict or a sequence of (min,max) pairs')
        return bounds

    def differential_evolution(self, max_iter: int = 20, disp: bool = True, tol: float = 0.01, atol: float = 0, 
                               popsize: int = 10, mutation: tuple = (0.5, 1), recombination: float = 0.65, 
                               updating: str = 'immediate', polish: bool = False):
        """Run SciPy differential_evolution using the class' attributes.
        Parameters:
        - max_iter: Maximum number of iterations for the optimizer (default 20).
        - disp: Whether to display convergence messages (default True).
        - tol: Relative tolerance for convergence (default 0.01).
        - atol: Absolute tolerance for convergence (default 0).
        - popsize: Population size multiplier (default 10).
        - mutation: Mutation constant or tuple (default (0.5, 1)).
        - recombination: Recombination constant (default 0.65).
        - updating: 'immediate' or 'deferred' strategy for updating the population (default 'immediate').
        - polish: Whether to perform a final local search using the L-BFGS-B algorithm (default False).
        Returns:
        - result: OptimizeResult object from SciPy differential_evolution.
        Note: for earlier stops, decrese max_iter and check self.iter_df for results. 
        Note: also increase atol to have, for the population at one iteration: std(costs) ≤ atol(costs) + tol(costs) * abs(mean(costs))!
        """
        # Reset iter_df for a fresh optimization run
        self.iter_df = pd.DataFrame()
        bounds = self._build_bounds()
        result = differential_evolution(
            self.cost_function_handler,
            bounds,
            x0=self.initial_guesses,
            disp=disp,
            tol=tol,
            atol=atol,
            maxiter=max_iter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            updating=updating,
            polish=polish,
        )
        return result

    def nelder_mead(self, max_iter: int = 500, return_all: bool = True, disp: bool = True, xatol: float = 1e-2, fatol: float = 1e-2):
        """Run SciPy Nelder-Mead (via minimize) using the class' attributes.
        Parameters:
        - max_iter: Maximum number of iterations for the optimizer (default 500).
        - return_all: Whether to return all solutions at each iteration (default True).
        - disp: Whether to display convergence messages (default True).
        - xatol: Absolute error in xopt between iterations to declare convergence (default 1e-2).
        - fatol: Absolute error in func(xopt) between iterations to declare convergence (default 1e-2).
        Returns:
        - result: OptimizeResult object from SciPy minimize.
        """
        # Reset iter_df for a fresh optimization run
        self.iter_df = pd.DataFrame()
        bounds = self._build_bounds()
        result = minimize(self.cost_function_handler, self.initial_guesses, bounds=bounds,
                            method='Nelder-Mead', options={'maxiter': max_iter, 'return_all': return_all, 'disp': disp, 'xatol': xatol, 'fatol': fatol})
        return result

    def save_iter_df(self, path: str):
        """Save the iteration DataFrame to an Excel file.

        Uses pandas.DataFrame.to_excel. For CSV, call `iter_df.to_csv`.
        """
        self.iter_df.to_excel(path, index=False)