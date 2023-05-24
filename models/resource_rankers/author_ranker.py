"""
Objects and methods for ranking academic resources based on authors.
"""
import numpy as np
import time
from models.collab_filtering import contextbased_cf as ccf, userbased_cf as ucf
from models.hyperparams import Hyperparams as hp
from models.resource import RankableResource, Resource
from models.resource_rankers.ranker import Ranker


class AuthorRanker(Ranker):
    @classmethod
    def _get_res_idx_dict(cls, resources):
        """
        Assigns a unique integer index to each resource and author.
        The indices remain the same if the input list is in the same order.
        The set of indices for resources and authors are separate.

        :param resources: The list of resources to extract from.
        :type resources: list[Resource]
        :return: The dictionary mapping from resources / authors to indices.
        :rtype: tuple[dict[Resource, int], dict[str, int]]
        """
        resource_corpus: list[Resource] = []
        author_corpus: list[str] = []

        for resource in resources:
            resource_corpus.append(resource)

            if resource.authors is None:
                continue
            for author in resource.authors:
                author_corpus.append(author)

        # Remove any duplicates from corpora.
        resource_corpus = list(dict.fromkeys(resource_corpus))
        author_corpus = list(dict.fromkeys(author_corpus))

        return (
            {r: i for i, r in enumerate(resource_corpus)},
            {r: i for i, r in enumerate(author_corpus)},
        )

    @classmethod
    def _get_rel_mat(cls, resource_idx_dict, author_idx_dict):
        """
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.

        :param resource_idx_dict: The resource-to-index dict.
        :type resource_idx_dict: dict[Resource, int]
        :param author_idx_dict: The author-to-index dict.
        :type author_idx_dict: dict[str, int]
        :return: The author relation matrix, where C[i][j] = 1 if i is authored by j.
        :rtype: np.ndarray
        """
        rel_mat = np.zeros((len(resource_idx_dict), len(author_idx_dict)))

        for resource, resource_idx in resource_idx_dict.items():
            if resource.authors is None:
                continue
            for author in resource.authors:
                author_idx = author_idx_dict[author]
                rel_mat[resource_idx][author_idx] = 1

        return rel_mat

    @classmethod
    def set_resource_rankings(
        cls,
        rankable_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks candidate resources from best to worst according to how similar
        their association matrix paper vectors are to targets (if using CCF).
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

        resource_idx_dict, author_idx_dict = cls._get_res_idx_dict(
            rankable_resources + target_resources
        )
        rel_mat = cls._get_rel_mat(resource_idx_dict, author_idx_dict)

        if cf_method == "userbased":
            # Use user-based collaborative filtering (UCF).
            sim_mat = ucf.get_similarity_matrix(rel_mat=rel_mat)
        if cf_method == "contextbased":
            # Use context-based collaborative filtering (CCF).
            ass_mat = ccf.get_association_matrix(
                rel_mat=rel_mat,
                user_idxs=resource_idx_dict.values(),
                cooc_prob_ts=hp.AUTHOR_COOC_PROB_TS
            )
            sim_mat = ccf.get_similarity_matrix(ass_mat=ass_mat)

        res_idx_pair_list = resource_idx_dict.items()
        target_idx_pair_list = [
            (r, i) for r, i in res_idx_pair_list if r in target_resources
        ]
        cand_idx_pair_list = [
            (r, i) for r, i in res_idx_pair_list if r in rankable_resources
        ]

        # Keep track of the similarity scores obtained by each candidate.
        sim_dict: dict[RankableResource, [float]] = {}
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

        # Assign the ranking position for each Resource object.
        for i, (res, sim) in enumerate(sim_list):
            if sim == 0.0:
                # In the case where many resources get a 0 similarity score,
                # they should all receive an equal ranking,
                # rather than be sorted in alphabetical order.
                res.author_based_ranking = len(sim_list)
            else:
                res.author_based_ranking = i + 1


if __name__ == "__main__":
    i1 = RankableResource({"title": "i1"})
    i2 = RankableResource({"title": "i2"})
    i3 = RankableResource({"title": "i3"})
    i4 = RankableResource({"title": "i4"})
    i5 = RankableResource({"title": "i5"})
    i1.authors = ["j1"]
    i2.authors = ["j1", "j2"]
    i3.authors = ["j1", "j2"]
    i4.authors = ["j2"]
    i5.authors = ["j1"]

    t1 = time.time()
    resource_idx_dict, author_idx_dict = AuthorRanker._get_res_idx_dict(
        [i1, i2, i3, i4, i5]
    )
    rel_mat = AuthorRanker._get_rel_mat(
        resource_idx_dict,
        author_idx_dict
    )
    t2 = time.time()
    print(f"author_ranker: Relation matrix:\n{np.transpose(rel_mat)}")
    print(f"author_ranker: Time taken to execute: {t2 - t1} seconds")

    t1 = time.time()
    AuthorRanker.set_resource_rankings(
        [i2, i4, i5],
        [i1, i3]
    )
    t2 = time.time()
    print("author_ranker: Ranked candidate resources:")
    for i in [i2, i4, i5]:
        print(f"\t[{i.author_based_ranking}]: {i.title}")
    print(f"author_ranker: Time taken to execute: {t2 - t1} seconds")
