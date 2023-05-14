"""
Objects and methods for ranking academic resources based on authors.
"""
import numpy as np
import time
from models import contextbased_cf as ccf
from models.hyperparams import Hyperparams as hp
from models.resource import Resource
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
    def get_ranked_cand_resources(
        cls,
        candidate_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks candidate resources from best to worst according to how similar
        their association matrix paper vectors are to the targets.
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279056.

        :param candidate_resources: The list of candidate resources to rank.
        :type candidate_resources: list[Resource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        :return: The list of citing resources, sorted by the CCF algorithm.
        :rtype: list[Resource]
        """
        resource_idx_dict, author_idx_dict = cls._get_res_idx_dict(
            candidate_resources + target_resources
        )
        rel_mat = cls._get_rel_mat(resource_idx_dict, author_idx_dict)
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
            (r, i) for r, i in res_idx_pair_list if r in candidate_resources
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
    i1 = Resource({"title": "i1"})
    i2 = Resource({"title": "i2"})
    i3 = Resource({"title": "i3"})
    i4 = Resource({"title": "i4"})
    i5 = Resource({"title": "i5"})
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
    ranked_candidates = AuthorRanker.get_ranked_cand_resources(
        [i2, i4, i5],
        [i1, i3]
    )
    t2 = time.time()
    print("author_ranker: Ranked candidate resources:")
    for i, ranked_candidate in enumerate(ranked_candidates):
        print(f"\t[{i + 1}]: {ranked_candidate.title}")
    print(f"author_ranker: Time taken to execute: {t2 - t1} seconds")
