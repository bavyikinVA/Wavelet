import numpy as np


def change_channels(v1, v2, data):
    def gram_schmidt(v1_, v2_):
        v1_ = np.array(v1_, dtype=np.float64)
        v2_ = np.array(v2_, dtype=np.float64)

        v1_normalize = v1_ / np.linalg.norm(v1_)

        v2_proj = np.dot(v2_, v1_normalize) * v1_normalize
        v2_orth = v2_ - v2_proj

        v2_orth_normalize = v2_orth / np.linalg.norm(v2_orth)

        return v1_normalize, v2_orth_normalize

    def cross_product(v1_, v2_):
        v3 = np.cross(v1_, v2_)
        v3_normalize = v3 / np.linalg.norm(v3)
        return v3_normalize

    data = np.array(data, dtype=np.float64)

    # Perform Gram-Schmidt
    v1_norm, v2_orth_norm = gram_schmidt(v1, v2)
    v3_norm = cross_product(v1_norm, v2_orth_norm)

    # Transformation matrix
    p = np.array([v1_norm, v2_orth_norm, v3_norm]).T
    print(p)
    # Transform image data
    data_transposed = np.transpose(data, (2, 0, 1))
    data_new = p @ data_transposed
    data_new = np.transpose(data_new, (1, 2, 0))

    return data_new