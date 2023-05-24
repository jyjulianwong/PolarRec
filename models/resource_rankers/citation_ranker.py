"""
Objects and methods for ranking academic resources based on citations.
"""
import numpy as np
import time
from models.citation_database_adapters.s2ag_adapter import S2agAdapter
from models.collab_filtering import contextbased_cf as ccf, userbased_cf as ucf
from models.hyperparams import Hyperparams as hp
from models.resource import RankableResource, Resource
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
        citing_resource_corpus: list[Resource] = []
        cited_resource_corpus: list[Resource] = []

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
    def set_resource_rankings(
        cls,
        rankable_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks citing resources by comparing the similarities of the paper
        vectors from the association matrix (if using CCF).
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.
        This function optionally accepts 1 additional keyword argument:
            ``cf_method: str``.

        :param rankable_resources: The list of resources to rank.
        :type rankable_resources: list[RankableResource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        """
        # Extract additional required keyword arguments.
        cf_method = kwargs.get("cf_method", hp.UNIVERSAL_CF_METHOD)

        citing_res_idx_dict, cited_res_idx_dict = cls._get_res_idx_dict(
            rankable_resources + target_resources
        )
        rel_mat = cls._get_rel_mat(citing_res_idx_dict, cited_res_idx_dict)

        if cf_method == "userbased":
            # Use user-based collaborative filtering (UCF).
            sim_mat = ucf.get_similarity_matrix(rel_mat=rel_mat)
        if cf_method == "contextbased":
            # Use context-based collaborative filtering (CCF).
            ass_mat = ccf.get_association_matrix(
                rel_mat=rel_mat,
                user_idxs=citing_res_idx_dict.values(),
                cooc_prob_ts=hp.CITATION_COOC_PROB_TS
            )
            sim_mat = ccf.get_similarity_matrix(ass_mat=ass_mat)

        citing_idx_pairs = citing_res_idx_dict.items()
        targ_citing_idx_pairs = [
            (r, i) for r, i in citing_idx_pairs if r in target_resources
        ]
        cand_citing_idx_pairs = [
            (r, i) for r, i in citing_idx_pairs if r in rankable_resources
        ]

        # Keep track of the similarity scores obtained by each candidate.
        sim_dict: dict[RankableResource, [float]] = {}
        for r, i in cand_citing_idx_pairs:
            for r0, i0 in targ_citing_idx_pairs:
                if r not in sim_dict:
                    sim_dict[r] = [sim_mat[i0][i]]
                else:
                    sim_dict[r].append(sim_mat[i0][i])

        # Calculate the mean similarity score for each candidate across targets.
        sim_list = [(r, sum(l) / len(l)) for r, l in sim_dict.items()]
        # Sort the candidates by their mean similarity score.
        sim_list = sorted(sim_list, key=lambda x: (x[1], x[0]), reverse=True)

        # Assign the ranking position for each Resource object.
        for i, (res, sim) in enumerate(sim_list):
            if sim == 0.0:
                # In the case where many resources get a 0 similarity score,
                # they should all receive an equal ranking,
                # rather than be sorted in alphabetical order.
                res.citation_based_ranking = len(sim_list)
            else:
                res.citation_based_ranking = i + 1


if __name__ == "__main__":
    print("\ncitation_ranker: Rank resources from abstract example")

    # Recreate the example from the paper that introduced this algorithm:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056
    i1 = RankableResource({"title": "i1"})
    i2 = RankableResource({"title": "i2"})
    i3 = RankableResource({"title": "i3"})
    i4 = RankableResource({"title": "i4"})
    i5 = RankableResource({"title": "i5"})
    j1 = RankableResource({"title": "j1"})
    j2 = RankableResource({"title": "j2"})
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
    CitationRanker.set_resource_rankings(
        [i2, i4, i5],
        [i1, i3]
    )
    t2 = time.time()
    print("citation_ranker: Ranked candidate resources:")
    for i in [i2, i4, i5]:
        print(f"\t[{i.citation_based_ranking}]: {i.title}")
    print(f"citation_ranker: Time taken to execute: {t2 - t1} seconds")

    print("\ncitation_ranker: Similarity matrix for real example with CCF...")
    print("... where the SegNet paper is the only common cited paper")

    target_data1 = {
        "authors": [],
        "title": "Texture-aware Network for Smoke Density Estimation",
        "year": 2022,
        "doi": "10.1109/VCIP56404.2022.10008826",
        "url": "https://ieeexplore.ieee.org/document/10008826"
    }
    target_resource1 = Resource(target_data1)
    target_data2 = {
        "authors": [],
        "title": "A Review on Early Diagnosis of Skin Cancer Detection Using Deep Learning Techniques",
        "year": 2022,
        "doi": "10.1109/ICCPC55978.2022.10072274",
        "url": "https://ieeexplore.ieee.org/document/10072274"
    }
    target_resource2 = Resource(target_data2)

    reference_dict = S2agAdapter.get_references(
        [target_resource1, target_resource2]
    )
    for resource, references in reference_dict.items():
        resource.references = references
        print(f"citation_ranker: len(references): {len(resource.references)}")

    citing_res_idx_dict, cited_res_idx_dict = CitationRanker._get_res_idx_dict(
        [target_resource1, target_resource2]
    )
    print(f"citation_ranker: len(Citing resources): {len(citing_res_idx_dict)}")
    print(f"citation_ranker: len(Cited resources):  {len(cited_res_idx_dict)}")
    rel_mat = CitationRanker._get_rel_mat(
        citing_res_idx_dict,
        cited_res_idx_dict
    )
    print(f"citation_ranker: Relation matrix:\n{rel_mat}")
    ass_mat = ccf.get_association_matrix(
        rel_mat=rel_mat,
        user_idxs=citing_res_idx_dict.values(),
        cooc_prob_ts=hp.CITATION_COOC_PROB_TS
    )
    print(f"citation_ranker: Association matrix:\n{ass_mat}")
    sim_mat = ccf.get_similarity_matrix(ass_mat=ass_mat)
    print(f"citation_ranker: Similarity matrix:\n{sim_mat}")
