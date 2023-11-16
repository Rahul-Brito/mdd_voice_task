import numpy as np
import pandas as pd

def create_group_contrast(design_matrix, covariate):
    contrast = np.zeros(design_matrix.shape[1])
    cov_index = design_matrix.columns.get_loc(covariate)
    contrast[cov_index] = 1
    return contrast