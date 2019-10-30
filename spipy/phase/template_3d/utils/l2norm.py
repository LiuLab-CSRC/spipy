import numpy as np

def l2norm(array1,array2):
    """Calculate sqrt ( sum |array1 - array2|^2 / sum|array1|^2 )."""
    tot = np.sum(np.abs(array1)**2)
    return np.sqrt(np.sum(np.abs(array1-array2)**2)/tot)
