import numpy as np
from typing import Sequence, Tuple, Dict, Optional, Union
from scipy.stats import truncnorm, gaussian_kde
from SALib.sample import saltelli, latin
import logging

def generate_parameter_samples(
    x_names: Sequence[str],
    bnds: Sequence[Tuple[float, float]],
    sampling_method: str,
    n_samples: int,
    param_distributions: Optional[Sequence[Tuple[float, float]]] = None,
    bootstrap_samples: Optional[Dict[str, np.ndarray]] = None,
    kde_bandwidth: Optional[Union[str, float]] = None,
    calc_second_order: bool = False,
    std_fraction: float = 0.15,
    random_state: Optional[int] = None,
    max_rejection_factor: float = 5.0,
    clip_outside: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Generate parameter sample matrix given bounds and a specified sampling strategy.

    Parameters
    ----------
    x_names : Sequence[str]
        Ordered list of parameter names.
    bnds : Sequence[Tuple[float,float]]
        Per-parameter (min, max) bounds tuples; same order as x_names.
    sampling_method : str
        One of:
          - 'saltelli'      : SALib Saltelli Sobol sequence.
          - 'lhs'           : Latin Hypercube (uniform).
          - 'monte_carlo'   : Truncated normal (need param_distributions or std_fraction).
          - 'kde'           : KDE-based sampling from bootstrap_samples.
          - 'empirical'     : Empirical resampling (with replacement) from bootstrap_samples.
    n_samples : int
        Target number of samples (final rows) for ALL methods.
        For Saltelli, this is the desired final total; the internal Saltelli base size is derived
        and the result is trimmed or expanded (never expanded beyond achievable) to match. Ideally n_samples/(2 * len(x_names) + 2) and n_samples/(len(x_names) + 2) is a power of 2!!
        It is thus adviced to set alpha and then n_samples = 2**alpha*(2 * len(x_names) + 2) or 2**alpha*(len(x_names) + 2).
    param_distributions : Sequence[Tuple[mean, std]], optional
        Per-parameter (mean, std) tuples used by 'monte_carlo'. If None, mean is midpoint
        of bounds and std = std_fraction * (upper - lower).
    bootstrap_samples : Dict[str, np.ndarray], optional
        Dict mapping parameter name -> 1D array of bootstrap samples (for 'kde' and 'empirical').
    kde_bandwidth : str | float, optional
        Bandwidth passed to gaussian_kde (e.g., 'scott', 'silverman', or numeric).
    calc_second_order : bool, default False
        Whether Saltelli should generate samples for second-order Sobol indices.
    std_fraction : float, default 0.15
        Fallback fraction of bound range used to compute std if param_distributions is None.
    random_state : int, optional
        Seed for reproducibility.
    max_rejection_factor : float, default 5.0
        For KDE sampling: maximum allowed ratio of draws to desired samples before abort.
    clip_outside : bool, default False
        If True, samples falling outside bounds (after all attempts) are clipped instead of causing errors.

    Returns
    -------
    Tuple[np.ndarray, Dict]
        Sample matrix of shape (n_samples, D) where D = len(x_names).
        Problem dictionary compatible with SALib.

    Raises
    ------
    ValueError / RuntimeError
        If inputs are inconsistent or a method cannot satisfy constraints.
    """
    # Set global seed for SALib methods (Saltelli, LHS)
    if random_state is not None:
        np.random.seed(random_state)

    rng = np.random.default_rng(random_state)
    D = len(x_names)

    if len(bnds) != D:
        raise ValueError("Length of bnds must equal length of x_names.")

    if param_distributions is not None and len(param_distributions) != D:
        raise ValueError("param_distributions must have same length as x_names.")

    sampling_method = sampling_method.lower()
    problem = {'num_vars': D, 'names': list(x_names), 'bounds': list(bnds)}

    # SALTELLI
    if sampling_method == 'saltelli':
        if calc_second_order:
            # Derive base size N from desired total: total ≈ N * (2D + 2) (if second-order)
            denom = (2 * D + 2)
        else:
            # Derive base size N from desired total: total ≈ N * (D + 2) (first-order only)
            denom = (D + 2)
        base_n = max(2, int(np.ceil(n_samples / denom)))
        logging.info(f"Derived base_n for Saltelli sampling: {base_n}")

        raw = saltelli.sample(problem, base_n, calc_second_order=calc_second_order)
        # Trim if overshoot
        if raw.shape[0] > n_samples:
            param_values = raw[:n_samples, :]
        else:
            param_values = raw
            if param_values.shape[0] < n_samples:
                # Cannot safely expand Saltelli sequence; warn via error unless user accepts fewer
                raise RuntimeError(
                    f"Saltelli produced {param_values.shape[0]} samples (< requested {n_samples}). "
                    f"Increase n_samples or accept the smaller set."
                )

    # LHS
    elif sampling_method == 'lhs':
        param_values = latin.sample(problem, n_samples)

    # MONTE CARLO (truncated normal)
    elif sampling_method == 'monte_carlo':
        param_values = np.zeros((n_samples, D))
        if param_distributions is None: # Not sure about this < ------------------------------------
            logging.info("param_distributions not provided; using bounds midpoints and std_fraction.")
            param_distributions = []
            for (low, high) in bnds:
                mean = (low + high) / 2.0
                std = (high - low) * std_fraction
                param_distributions.append((mean, std))

        for i, (low, high) in enumerate(bnds):
            mean, std = param_distributions[i]
            if std <= 0:
                raise ValueError(f"Std must be > 0 for parameter '{x_names[i]}'.")
            a = (low - mean) / std
            b = (high - mean) / std
            param_values[:, i] = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples, random_state=rng) # Not sure about this < ------------------------------------

    # KDE FROM BOOTSTRAP
    elif sampling_method == 'kde':
        if bootstrap_samples is None:
            raise ValueError("bootstrap_samples required for 'kde' method.")
        param_values = np.zeros((n_samples, D))
        for i, name in enumerate(x_names):
            if name not in bootstrap_samples:
                raise ValueError(f"Missing bootstrap samples for '{name}'.")
            data = np.asarray(bootstrap_samples[name])
            if data.ndim != 1 or data.size < 10:
                raise ValueError(f"Bootstrap array for '{name}' must be 1D with >=10 samples.")
            kde = gaussian_kde(data, bw_method=kde_bandwidth)
            low, high = bnds[i]

            collected = []
            max_draws = int(n_samples * max_rejection_factor)
            draws = 0
            while len(collected) < n_samples and draws < max_draws:
                batch_size = min(64, n_samples - len(collected))
                batch = kde.resample(size=batch_size).ravel()
                for val in batch:
                    if low <= val <= high:
                        collected.append(val)
                        if len(collected) == n_samples:
                            break
                draws += batch_size

            if len(collected) < n_samples:
                if clip_outside:
                    remaining = n_samples - len(collected)
                    extra = kde.resample(size=remaining).ravel()
                    extra = np.clip(extra, low, high)
                    collected.extend(extra.tolist())
                else:
                    raise RuntimeError(
                        f"KDE rejection sampling for '{name}' collected {len(collected)}/{n_samples}. "
                        f"Increase max_rejection_factor or enable clip_outside."
                    )
            param_values[:, i] = np.array(collected)

    # EMPIRICAL RESAMPLING
    elif sampling_method == 'empirical':
        if bootstrap_samples is None:
            raise ValueError("bootstrap_samples required for 'empirical' method.")
        param_values = np.zeros((n_samples, D))
        for i, name in enumerate(x_names):
            if name not in bootstrap_samples:
                raise ValueError(f"Missing bootstrap samples for '{name}'.")
            data = np.asarray(bootstrap_samples[name])
            low, high = bnds[i]
            in_bounds = data[(data >= low) & (data <= high)]
            if in_bounds.size == 0:
                if clip_outside:
                    in_bounds = np.clip(data, low, high)
                else:
                    raise RuntimeError(f"No bootstrap samples within bounds for '{name}'.")
            param_values[:, i] = rng.choice(in_bounds, size=n_samples, replace=True)

    else:
        raise ValueError(
            f"Unknown sampling_method '{sampling_method}'. "
            "Choose among 'saltelli', 'lhs', 'monte_carlo', 'kde', 'empirical'."
        )

    # Final sanity clip (optional safeguard)
    for i, (low, high) in enumerate(bnds):
        out_of_bounds = (param_values[:, i] < low) | (param_values[:, i] > high)
        if np.any(out_of_bounds):
            if clip_outside:
                param_values[:, i] = np.clip(param_values[:, i], low, high)
            else:
                raise RuntimeError(
                    f"Sampling produced values outside bounds for '{x_names[i]}'. "
                    "Enable clip_outside=True to force clipping."
                )

    return param_values, problem