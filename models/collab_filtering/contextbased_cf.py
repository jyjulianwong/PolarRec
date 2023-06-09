"""
Objects and methods for ranking academic resources via context-based
collaborative filtering (CCF). For the original implementation, see
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
"""
import numpy as np
import scipy.special as sc
import time


def get_cooccurrence_prob(rel_vec1, rel_vec2):
    """
    See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    Referencing Section III (A).

    :param rel_vec1: A user vector from the relation matrix.
    :type rel_vec1: np.ndarray
    :param rel_vec2: A user vector from the relation matrix.
    :type rel_vec2: np.ndarray
    :return: The co-occurrence probability value.
    :rtype: float
    """
    # Calculate the raw values of the contingency table.
    sum_vec = np.add(rel_vec1, rel_vec2)
    n11 = np.shape(sum_vec[sum_vec == 2])[-1]
    n22 = np.shape(sum_vec[sum_vec == 0])[-1]
    sub_vec = np.subtract(rel_vec1, rel_vec2)
    n12 = np.shape(sub_vec[sub_vec == 1])[-1]
    n21 = np.shape(sub_vec[sub_vec == -1])[-1]
    # Calculate the derivative values from the contingency table.
    r1 = n11 + n12
    r2 = n21 + n22
    c1 = n11 + n21
    c2 = n12 + n22
    n = c1 + c2

    # Calculate the number of degrees of freedom of the contingency table.
    v = 1
    # Calculate the chi-squared value.
    x = (abs(n11 * n22 - n12 * n21) - n / 2) ** 2 / max(r1 * r2 * c1 * c2, 1)

    # Calculate the co-occurrence probability,
    # equal to the CDF of the chi-squared distribution,
    # using the gamma function and regularised upper incomplete gamma function.
    prob = 1 - (sc.gammaincc(v / 2, x / 2) / sc.gamma(v / 2))
    return prob


def get_association_matrix(rel_mat, user_idxs, cooc_prob_ts=0.1):
    """
    See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    Referencing Section III (A).

    :param rel_mat: The relation matrix.
    :type rel_mat: np.ndarray
    :param user_idxs: The list of indices referring to "users".
    :type user_idxs: list[int]
    :param cooc_prob_ts: The co-occurrence probability threshold.
    :type cooc_prob_ts: float
    :return: The association matrix, where each row is a user vector.
    :rtype: np.ndarray
    """
    # FIXME: Assumes user_idxs is a contiguous list of indices starting from 0.
    ass_mat = np.zeros((len(user_idxs), len(user_idxs)))

    for i1 in user_idxs:
        for i2 in user_idxs:
            if i1 == i2:
                continue

            rel_vec1 = rel_mat[i1]
            rel_vec2 = rel_mat[i2]
            cooc_prob = get_cooccurrence_prob(rel_vec1, rel_vec2)
            if cooc_prob > cooc_prob_ts:
                ass_mat[i1][i2] = 1

    return ass_mat


def get_association_vec_sim(ass_vec1, ass_vec2):
    """
    Calculates the cosine similarity between two association matrix user
    vectors.
    See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    Referencing Section III (A).

    :param ass_vec1: A user vector from the association matrix.
    :type ass_vec1: np.ndarray
    :param ass_vec2: A user vector from the association matrix.
    :type ass_vec2: np.ndarray
    :return: The cosine similarity between two association matrix user vectors.
    :rtype: float
    """
    num = np.dot(ass_vec1, ass_vec2)
    den = np.linalg.norm(ass_vec1) * np.linalg.norm(ass_vec2)
    # Set minimum denominator to avoid division-by-zero.
    return num / max(den, 1e-9)


def get_similarity_matrix(ass_mat):
    """
    See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    Referencing Section III (B).

    :param ass_mat: The association matrix.
    :type ass_mat: np.ndarray
    :return: A matrix with the similarities between every pair of users.
    :rtype: np.ndarray
    """
    sim_mat = np.zeros(ass_mat.shape)

    for i1 in range(ass_mat.shape[0]):
        for i2 in range(i1, ass_mat.shape[1]):
            if i1 == i2:
                sim_mat[i1][i2] = 1
                continue

            ass_vec1 = ass_mat[i1]
            ass_vec2 = ass_mat[i2]
            sim = get_association_vec_sim(ass_vec1, ass_vec2)
            sim_mat[i1][i2] = sim
            sim_mat[i2][i1] = sim

    return sim_mat


def get_score_matrix(rel_mat, user_idxs, item_idxs, target_user_idxs):
    """
    The indices in ``target_user_idxs`` must be included in ``user_idxs``.
    See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
    Referencing Section III (C).

    :param rel_mat: The relation matrix.
    :type rel_mat: np.ndarray
    :param user_idxs: The list of indices referring to "users".
    :type user_idxs: list[int]
    :param item_idxs: The list of indices referring to "items".
    :type item_idxs: list[int]
    :param target_user_idxs: The list of users whose scores will be calculated.
    :type target_user_idxs: list[int]
    :return: The utility scores for each user and item pair.
    :rtype: np.ndarray
    """
    ass_mat = get_association_matrix(rel_mat, user_idxs)
    sim_mat = get_similarity_matrix(ass_mat)

    # FIXME: Assumes user_idxs is a contiguous list of indices starting from 0.
    # FIXME: Assumes item_idxs is a contiguous list of indices starting from 0.
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
    ass_mat = get_association_matrix(rel_mat, user_idxs)
    t2 = time.time()
    print(f"contextbased_cf: Association matrix:\n{ass_mat}")
    print(f"contextbased_cf: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    sim_mat = get_similarity_matrix(ass_mat)
    t2 = time.time()
    print(f"contextbased_cf: Similarity matrix:\n{sim_mat}")
    print(f"contextbased_cf: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    score_mat = get_score_matrix(
        rel_mat,
        user_idxs,
        item_idxs,
        [0, 2]
    )
    t2 = time.time()
    print(f"contextbased_cf: Citation score matrix:\n{np.transpose(score_mat)}")
    print(f"contextbased_cf: Time taken to execute: {t2 - t1} seconds")
