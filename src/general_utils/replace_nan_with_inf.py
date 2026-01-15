# Function that strongly penalizes scenario evaluations that return values containing nan (unfeasible in some way)
import numpy as np

def replace_nan_with_inf(d, penaltyterm):
    for key, value in d.items():
        if isinstance(value, float) and np.isnan(value):
            d[key] = penaltyterm
    return d