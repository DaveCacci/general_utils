import scipy.stats as stats

def T_test(estimate: float, hypothesized_value: float, standard_error: float, degrees_of_freedom: int, alpha: float = 0.05, alternative: str = 'two-sided'):
    """
    Perform a t-test to check the statistical significance of an estimated parameter.
    
    Parameters:
    - estimate: The estimated parameter value (e.g., regression coefficient).
    - hypothesized_value: The hypothesized value (often 0).
    - standard_error: The standard error of the estimate.
    - alpha: Significance level (default is 0.05).
    - alternative: Specifies the alternative hypothesis ('two-sided', 'greater', or 'less').
    
    Returns:
    - t_statistic: The calculated t-statistic.
    - p_value: The p-value associated with the t-statistic.
    - is_significant: Boolean indicating if the estimate is statistically significant.
    """
    
    # Calculate the t-statistic
    t_statistic = (estimate - hypothesized_value) / standard_error
    
    # Calculate the p-value based on the alternative hypothesis
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=degrees_of_freedom))
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_statistic, df=degrees_of_freedom)
    elif alternative == 'less':
        p_value = stats.t.cdf(t_statistic, df=degrees_of_freedom)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    # Determine if the result is statistically significant
    is_significant = p_value < alpha
    
    return t_statistic, p_value, is_significant