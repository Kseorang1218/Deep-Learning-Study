# features.py

import numpy as np

def rms(data):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    rms = np.sqrt(np.mean(data**2))
    return rms
