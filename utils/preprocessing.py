import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

SPECIAL_MISSING = [-1, -9, 9, 99, 999, 9999]



def clean_numeric(series, special_missing=SPECIAL_MISSING):
    s = pd.to_numeric(series, errors="coerce")
    return s.replace(special_missing, np.nan)


def minmax_normalize(series):
    s = pd.to_numeric(series, errors="coerce")

    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)

    return (s - s.min()) / (s.max() - s.min())