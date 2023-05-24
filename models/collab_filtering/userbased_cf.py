import math
import numpy as np
import time


def get_relation_vector_similarity(rel_vec1, rel_vec2):
    """
    Calculates the similarity between two relation matrix user vectors.

    :param rel_vec1: A user vector from the relation matrix.
    :type rel_vec1: np.ndarray
    :param rel_vec2: A user vector from the relation matrix.
    :type rel_vec2: np.ndarray
    :return: The cosine similarity between two relation matrix user vectors.
    :rtype: float
    """
    num = np.dot(rel_vec1, rel_vec2)
    den = np.linalg.norm(rel_vec1) * np.linalg.norm(rel_vec2)
    # Set minimum denominator to avoid division-by-zero.
    return num / max(den, 1e-9)


def get_similarity_matrix(rel_mat):
    """
    :param rel_mat: The relation matrix.
    :type rel_mat: np.ndarray
    :return: The similarity between every pair of citing papers.
    :rtype: np.ndarray
    """
    sim_mat = np.zeros((rel_mat.shape[0], rel_mat.shape[0]))

    for i1 in range(sim_mat.shape[0]):
        for i2 in range(i1, sim_mat.shape[1]):
            if i1 == i2:
                sim_mat[i1][i2] = 1
                continue

            rel_vec1 = rel_mat[i1]
            rel_vec2 = rel_mat[i2]
            sim = get_relation_vector_similarity(rel_vec1, rel_vec2)
            sim_mat[i1][i2] = sim
            sim_mat[i2][i1] = sim

    return sim_mat


def get_score_matrix(rel_mat, user_idxs, item_idxs, target_user_idxs):
    """
    The indices in ``target_user_idxs`` must be included in ``user_idxs``.

    :param rel_mat: The relation matrix.
    :type rel_mat: np.ndarray
    :param user_idxs: The list of indices referring to "users".
    :type user_idxs: list[int]
    :param item_idxs: The list of indices referring to "items".
    :type item_idxs: list[int]
    :param target_user_idxs: The list of users whose scores will be calculated.
    :type target_user_idxs: list[int]
    :return: The scores between each user and item pair.
    :rtype: np.ndarray
    """
    sim_mat = get_similarity_matrix(rel_mat)

    # TODO: Assumes user_idxs is a contiguous list of indices starting from 0.
    # TODO: Assumes item_idxs is a contiguous list of indices starting from 0.
    score_mat = np.zeros((len(user_idxs), len(item_idxs)))
    for i0 in target_user_idxs:
        for j in item_idxs:
            weighted_score_sum = 0
            for i in user_idxs:
                if i0 != i:
                    weighted_score_sum += sim_mat[i0][i] * rel_mat[i][j]
            sim_sum = np.sum(sim_mat[i0]) - sim_mat[i0][i0]
            # Set minimum denominator to avoid division-by-zero.
            score_mat[i0][j] = weighted_score_sum / max(sim_sum, 1e-9)

    return score_mat

if __name__ == "__main__":
    # Recreate the example from the paper that introduced this algorithm:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056
    # In this implementation, each row of the relation matrix refers to a user,
    # and each column refers to an item.
    rel_mat = np.array([
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 1],
        [1, 0]
    ])
    user_idxs = [0, 1, 2, 3, 4]
    item_idxs = [0, 1]

    t1 = time.time()
    sim_mat = get_similarity_matrix(rel_mat)
    t2 = time.time()
    print(f"userbased_cf: Similarity matrix:\n{sim_mat}")
    print(f"userbased_cf: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    score_mat = get_score_matrix(
        rel_mat,
        user_idxs,
        item_idxs,
        [0, 2]
    )
    t2 = time.time()
    print(f"userbased_cf: Citation score matrix:\n{np.transpose(score_mat)}")
    print(f"userbased_cf: Time taken to execute: {t2 - t1} seconds")
