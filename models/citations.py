"""
Objects and methods for ranking academic resources based on citations.
"""
import math
import numpy as np
import time
from models.resource import Resource

# The threshold value above which two resources have significant co-occurrence.
CITING_RES_COOC_PROB_TS = 0.1


def get_resource_index_dict(citing_resources):
    """
    Assigns a unique integer index to each citing and cited resource.
    The indices remain the same as long as the input list is in the same order.
    The set of indices for citing and cited resources are separate.

    :param citing_resources: The list of resources to extract from.
    :type citing_resources: list[Resource]
    :return: The dictionary mapping from resources to unique indices.
    :rtype: tuple[dict[Resource, int], dict[Resource, int]]
    """
    citing_resource_corpus = []
    cited_resource_corpus = []

    for citing_resource in citing_resources:
        citing_resource_corpus.append(citing_resource)

        if citing_resource.references is None:
            continue
        for cited_resource in citing_resource.references:
            cited_resource_corpus.append(cited_resource)

    citing_resource_corpus = list(dict.fromkeys(citing_resource_corpus))
    cited_resource_corpus = list(dict.fromkeys(cited_resource_corpus))

    return (
        {r: i for i, r in enumerate(citing_resource_corpus)},
        {r: i for i, r in enumerate(cited_resource_corpus)},
    )


def get_citation_relation_matrix(citing_res_idx_dict, cited_res_idx_dict):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056

    :param citing_res_idx_dict: The resource-to-index dict for citing resources.
    :type citing_res_idx_dict: dict[Resource, int]
    :param cited_res_idx_dict: The resource-to-index dict for cited resources.
    :type cited_res_idx_dict: dict[Resource, int]
    :return: The citation relation matrix, where C[i][j] = 1 if i cites j.
    :rtype: np.array
    """
    rel_mat = np.zeros((len(citing_res_idx_dict), len(cited_res_idx_dict)))

    for citing_resource, citing_res_idx in citing_res_idx_dict.items():
        if citing_resource.references is None:
            continue
        for cited_resource in citing_resource.references:
            cited_res_idx = cited_res_idx_dict[cited_resource]
            rel_mat[citing_res_idx][cited_res_idx] = 1

    return rel_mat


def get_cooccurence_prob(rel_vec1, rel_vec2):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056

    :param rel_vec1: A paper vector from the citation relation matrix.
    :type rel_vec1: np.array
    :param rel_vec2: A paper vector from the citation relation matrix.
    :type rel_vec2: np.array
    :return: The co-occurrence probability value.
    :rtype: float
    """
    sum_vec = np.add(rel_vec1, rel_vec2)
    n11 = np.shape(sum_vec[sum_vec == 2])[-1]
    n22 = np.shape(sum_vec[sum_vec == 0])[-1]
    sub_vec = np.subtract(rel_vec1, rel_vec2)
    n12 = np.shape(sub_vec[sub_vec == 1])[-1]
    n21 = np.shape(sub_vec[sub_vec == -1])[-1]

    r1 = n11 + n12
    r2 = n21 + n22
    c1 = n11 + n21
    c2 = n12 + n22
    n = c1 + c2

    chi_sqd = (abs(n11 * n22 - n12 * n21) - n / 2) ** 2
    chi_sqd /= max(r1 * r2 * c1 * c2, 1)

    prob = (math.e ** -0.5) * (chi_sqd ** 2)
    prob /= 2 * ((2 * math.pi) ** 0.5)
    return prob


def get_association_matrix(citing_res_idx_dict, cited_res_idx_dict):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056

    :param citing_res_idx_dict: The resource-to-index dict for citing resources.
    :type citing_res_idx_dict: dict[Resource, int]
    :param cited_res_idx_dict: The resource-to-index dict for cited resources.
    :type cited_res_idx_dict: dict[Resource, int]
    :return: The association matrix, where each row is a paper vector.
    :rtype: np.array
    """
    rel_mat = get_citation_relation_matrix(
        citing_res_idx_dict,
        cited_res_idx_dict
    )

    ass_mat = np.zeros((len(citing_res_idx_dict), len(citing_res_idx_dict)))
    for r1, i1 in citing_res_idx_dict.items():
        for r2, i2 in citing_res_idx_dict.items():
            if i1 == i2:
                continue

            rel_vec1 = rel_mat[i1]
            rel_vec2 = rel_mat[i2]
            cooc_prob = get_cooccurence_prob(rel_vec1, rel_vec2)
            if cooc_prob > CITING_RES_COOC_PROB_TS:
                ass_mat[i1][i2] = 1

    return ass_mat


def get_ass_vec_sim(ass_vec1, ass_vec2):
    """
    Calculates the similarity between two association matrix paper vectors.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056

    :param ass_vec1: A paper vector from the association matrix.
    :type ass_vec1: np.array
    :param ass_vec2: A paper vector from the association matrix.
    :type ass_vec2: np.array
    :return: The cosine similarity between two association matrix paper vectors.
    :rtype: float
    """
    num = np.dot(ass_vec1, ass_vec2)
    den = np.linalg.norm(ass_vec1) * np.linalg.norm(ass_vec2)
    return num / den


def get_similarity_matrix(ass_mat):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056

    :param ass_mat: The association matrix.
    :type ass_mat: np.array
    :return: The similarity between every pair of citing papers.
    :rtype: np.array
    """
    sim_mat = np.zeros(ass_mat.shape)

    for i1 in range(ass_mat.shape[0]):
        for i2 in range(i1, ass_mat.shape[1]):
            if i1 == i2:
                sim_mat[i1][i2] = 1
                continue

            ass_vec1 = ass_mat[i1]
            ass_vec2 = ass_mat[i2]
            sim = get_ass_vec_sim(ass_vec1, ass_vec2)
            sim_mat[i1][i2] = sim
            sim_mat[i2][i1] = sim

    return sim_mat


def get_citation_score_matrix(
    citing_res_idx_dict,
    cited_res_idx_dict,
    target_resources
):
    """
    The resources in ``target_resources`` must be included in
    ``citing_res_idx_dict``.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056

    :param citing_res_idx_dict: The resource-to-index dict for citing resources.
    :type citing_res_idx_dict: dict[Resource, int]
    :param cited_res_idx_dict: The resource-to-index dict for cited resources.
    :type cited_res_idx_dict: dict[Resource, int]
    :param target_resources: The list of resources to get scores of.
    :type target_resources: list[Resource]
    :return: The citation scores between each citing and cited paper pair.
    :rtype: np.array
    """
    rel_mat = get_citation_relation_matrix(
        citing_res_idx_dict,
        cited_res_idx_dict
    )
    ass_mat = get_association_matrix(citing_res_idx_dict, cited_res_idx_dict)
    sim_mat = get_similarity_matrix(ass_mat)

    score_mat = np.zeros((len(citing_res_idx_dict), len(cited_res_idx_dict)))
    for target_resource in target_resources:
        i0 = citing_res_idx_dict[target_resource]
        for _, j in cited_res_idx_dict.items():
            weighted_score_sum = 0
            for _, i in citing_res_idx_dict.items():
                if i0 != i:
                    weighted_score_sum += sim_mat[i0][i] * rel_mat[i][j]
            sim_sum = np.sum(sim_mat[i0]) - sim_mat[i0][i0]
            # Set minimum denominator to avoid division-by-zero.
            score_mat[i0][j] = weighted_score_sum / max(sim_sum, 1e-9)

    return score_mat


if __name__ == "__main__":
    # Recreate the example from the paper that introduced this algorithm:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056
    i1 = Resource({"title": "i1"})
    i2 = Resource({"title": "i2"})
    i3 = Resource({"title": "i3"})
    i4 = Resource({"title": "i4"})
    i5 = Resource({"title": "i5"})
    j1 = Resource({"title": "j1"})
    j2 = Resource({"title": "j2"})
    i1.references = [j1]
    i2.references = [j1, j2]
    i3.references = [j1, j2]
    i4.references = [j2]
    i5.references = [j1]

    t1 = time.time()
    citing_res_idx_dict, cited_res_idx_dict = get_resource_index_dict(
        [i1, i2, i3, i4, i5]
    )
    rel_mat = get_citation_relation_matrix(
        citing_res_idx_dict,
        cited_res_idx_dict
    )
    t2 = time.time()
    print(f"citations: Citation relation matrix:\n{np.transpose(rel_mat)}")
    print(f"citations: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    ass_mat = get_association_matrix(citing_res_idx_dict, cited_res_idx_dict)
    t2 = time.time()
    print(f"citations: Association matrix:\n{ass_mat}")
    print(f"citations: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    sim_mat = get_similarity_matrix(ass_mat)
    t2 = time.time()
    print(f"citations: Similarity matrix:\n{sim_mat}")
    print(f"citations: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    score_mat = get_citation_score_matrix(
        citing_res_idx_dict,
        cited_res_idx_dict,
        [i1, i3]
    )
    t2 = time.time()
    print(f"citations: Citation score matrix:\n{np.transpose(score_mat)}")
    print(f"citations: Time taken to execute: {t2 - t1} seconds")
