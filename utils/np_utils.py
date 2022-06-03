import numpy as np


def one_hot_embed(x, num_categories):
    assert (np.min(x) >= 0) and (np.max(x) < num_categories)
    assert len(x.shape) == 1
    if num_categories > 2:
        embedding = np.zeros((len(x), num_categories))
        embedding[np.arange(len(x)), x] = 1.0
        return embedding
    else:
        return x.reshape(-1, 1) * 1.0


def is_dicrete_vector_np(x):
    try:
        assert len(x.squeeze().shape) == 1
        assert x.dtype == "int"
        assert x.min() >= 0
        return True
    except AssertionError:
        raise False

