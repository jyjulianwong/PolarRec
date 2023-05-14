"""
Citation database adapter interface.
"""
from models.resource import Resource


class Adapter:
    """
    Collects citation and reference data for academic resources.
    """
    # Used by APIs to identify the calling application as part of etiquette.
    APP_URL = "https://github.com/jyjulianwong/PolarRec"
    APP_MAILTO = "jyw19@ic.ac.uk"
    MAX_SEARCH_RESULTS = 10

    @classmethod
    def get_citation_count(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The number of times the resource had been cited previously.
        :rtype: int
        """
        pass

    @classmethod
    def get_references(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The reference resources that the target resource has used.
        :rtype: list[Resource]
        """
        pass

    @classmethod
    def get_references_in_batches(cls, resources):
        """
        Searches for references for each resource in batches for better
        efficiency.
        Groups resources that have DOIs together, and searches for their
        references in a single API call.
        Resources that do not have DOIs have to be searched independently.

        :param resources: The list of target resources.
        :type resources: list[Resource]
        :return: The reference resources that each target resource has used.
        :rtype: dict[Resource, list[Resource]]
        """
        pass