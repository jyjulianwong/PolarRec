"""
Resource ranker interface.
"""
from models.resource import RankableResource, Resource


class Ranker:
    """
    Ranks candidate academic resources with some specialised algorithm.
    """

    @classmethod
    def set_resource_rankings(
        cls,
        rankable_resources,
        target_resources,
        **kwargs
    ):
        """
        Ranks candidate resources from best to worst with some specialised
        algorithm, and returns them as a sorted list.

        :param rankable_resources: The list of resources to rank.
        :type rankable_resources: list[RankableResource]
        :param target_resources: The list of target resources to base ranking on.
        :type target_resources: list[Resource]
        """
        pass
