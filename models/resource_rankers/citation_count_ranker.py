"""
Objects and methods for ranking academic resources based on citation counts.
"""
from models.resource import RankableResource
from models.resource_rankers.ranker import Ranker


class CitationCountRanker(Ranker):
    @classmethod
    def set_resource_rankings(
        cls,
        rankable_resources,
        target_resources,
        **kwargs
    ):
        # The list of pairs of the resource's citation counts and the resource.
        # Have the citation count be the first element for sorting purposes.
        cit_count_list: list[tuple[int, RankableResource]] = []
        for resource in rankable_resources:
            if resource.citation_count is None:
                # Set the citation count as 0 if data is not available.
                cit_count_list.append((0, resource))
            else:
                cit_count_list.append((resource.citation_count, resource))
        # Sort the candidates by their citation counts.
        cit_count_list = sorted(cit_count_list, reverse=True)

        # Assign the ranking position for each Resource object.
        for i, (count, res) in enumerate(cit_count_list):
            if count == 0:
                # In the case where many resources get a 0 citation count,
                # they should all receive an equal ranking,
                # rather than be sorted in alphabetical order.
                res.citation_count_ranking = len(cit_count_list)
            else:
                res.citation_count_ranking = i + 1
