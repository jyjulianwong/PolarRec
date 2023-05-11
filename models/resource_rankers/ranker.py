"""
Resource ranker interface.
"""


class Ranker:
    """
    Ranks candidate academic resources with some specialised algorithm.
    """

    @classmethod
    def get_ranked_cand_resources(
        cls,
        candidate_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks candidate resources from best to worst with some specialised
        algorithm, and returns them as a sorted list.

        :param candidate_resources: The list of candidate resources to rank.
        :type candidate_resources: list[Resource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        :return: The list of candidate resources, sorted by ranking algorithm.
        :rtype: list[Resource]
        """
        pass
