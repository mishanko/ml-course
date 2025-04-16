import itertools

import numpy as np

# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________

def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # YOUR CODE HERE
    matches = []
    
    out = np.linalg.norm(des1[:, None, :] - des2[None, :, :], axis=-1)

    closest_to_first = np.argmin(out, axis=1)
    closest_to_second = np.argmin(out, axis=0)

    iterator = itertools.product(range(des1.shape[0]), range(des2.shape[0]))
    for i, j in iterator:
        if closest_to_first[i] == j and closest_to_second[j] == i:
            matches.append(DummyMatch(i, j, out[i, j]))

    sorted_matches = sorted(matches, key = lambda x: x.distance)
    return sorted_matches