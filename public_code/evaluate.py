import numpy as np

def mean_squared_error(pred, intercept, feature_matrix, response_vector) -> float:
    n, d = np.shape(feature_matrix)
    assert np.shape(pred) == (d,)
    assert np.shape(response_vector) == (n,)
    predictions = feature_matrix @ pred + np.ones(n) * intercept
    return (np.linalg.norm(predictions - response_vector) ** 2) / n