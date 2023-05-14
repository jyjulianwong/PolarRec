"""
Objects and methods for ranking academic resources based on citations.
"""
import numpy as np
import time
from models import contextbased_cf as ccf
from models.hyperparams import Hyperparams as hp
from models.resource import Resource
from models.resource_rankers.ranker import Ranker


class CitationRanker(Ranker):
    @classmethod
    def _get_res_idx_dict(cls, citing_resources):
        """
        Assigns a unique integer index to each citing and cited resource.
        The indices remain the same if the input list is in the same order.
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

        # Remove any duplicates from corpora.
        citing_resource_corpus = list(dict.fromkeys(citing_resource_corpus))
        cited_resource_corpus = list(dict.fromkeys(cited_resource_corpus))

        return (
            {r: i for i, r in enumerate(citing_resource_corpus)},
            {r: i for i, r in enumerate(cited_resource_corpus)},
        )

    @classmethod
    def _get_rel_mat(cls, citing_res_idx_dict, cited_res_idx_dict):
        """
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.

        :param citing_res_idx_dict: The resource-to-index dict for citing resources.
        :type citing_res_idx_dict: dict[Resource, int]
        :param cited_res_idx_dict: The resource-to-index dict for cited resources.
        :type cited_res_idx_dict: dict[Resource, int]
        :return: The citation relation matrix, where C[i][j] = 1 if i cites j.
        :rtype: np.ndarray
        """
        rel_mat = np.zeros((len(citing_res_idx_dict), len(cited_res_idx_dict)))

        for citing_resource, citing_res_idx in citing_res_idx_dict.items():
            if citing_resource.references is None:
                continue
            for cited_resource in citing_resource.references:
                cited_res_idx = cited_res_idx_dict[cited_resource]
                rel_mat[citing_res_idx][cited_res_idx] = 1

        return rel_mat

    @classmethod
    def get_ranked_cand_resources(
        cls,
        candidate_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks citing resources by comparing the similarities of the paper
        vectors from the association matrix.
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.

        :param candidate_resources: The list of candidate resources to rank.
        :type candidate_resources: list[Resource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        :return: The list of citing resources, sorted by the CCF algorithm.
        :rtype: list[Resource]
        """
        citing_res_idx_dict, cited_res_idx_dict = cls._get_res_idx_dict(
            candidate_resources + target_resources
        )
        rel_mat = cls._get_rel_mat(citing_res_idx_dict, cited_res_idx_dict)
        ass_mat = ccf.get_association_matrix(
            rel_mat=rel_mat,
            user_idxs=citing_res_idx_dict.values(),
            cooc_prob_ts=hp.CITATION_COOC_PROB_TS
        )
        sim_mat = ccf.get_similarity_matrix(ass_mat=ass_mat)

        citing_idx_pair_list = citing_res_idx_dict.items()
        target_idx_pair_list = [
            (r, i) for r, i in citing_idx_pair_list if r in target_resources
        ]
        cand_idx_pair_list = [
            (r, i) for r, i in citing_idx_pair_list if r in candidate_resources
        ]

        # Keep track of the similarity scores obtained by each candidate.
        sim_dict: dict[Resource, [float]] = {}
        for r, i in cand_idx_pair_list:
            for r0, i0 in target_idx_pair_list:
                if r not in sim_dict:
                    sim_dict[r] = [sim_mat[i0][i]]
                else:
                    sim_dict[r].append(sim_mat[i0][i])

        # Calculate the mean similarity score for each candidate across targets.
        sim_list = [(r, sum(l) / len(l)) for r, l in sim_dict.items()]
        # Sort the candidates by their mean similarity score.
        sim_list = sorted(sim_list, key=lambda x: (x[1], x[0]), reverse=True)
        return [r for r, s in sim_list]


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
    citing_res_idx_dict, cited_res_idx_dict = CitationRanker._get_res_idx_dict(
        [i1, i2, i3, i4, i5]
    )
    rel_mat = CitationRanker._get_rel_mat(
        citing_res_idx_dict,
        cited_res_idx_dict
    )
    t2 = time.time()
    print(f"citation_ranker: Relation matrix:\n{np.transpose(rel_mat)}")
    print(f"citation_ranker: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    ranked_candidates = CitationRanker.get_ranked_cand_resources(
        [i2, i4, i5],
        [i1, i3]
    )
    t2 = time.time()
    print("citation_ranker: Ranked candidate resources:")
    for i, ranked_candidate in enumerate(ranked_candidates):
        print(f"\t[{i + 1}]: {ranked_candidate.title}")
    print(f"citation_ranker: Time taken to execute: {t2 - t1} seconds")
