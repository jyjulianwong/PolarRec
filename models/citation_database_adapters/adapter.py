"""
Citation database adapter interface.
"""
from models.hyperparams import Hyperparams as hp
from models.resource import Resource


class Adapter:
    """
    Collects citation and reference data for academic resources.
    """
    # Used by APIs to identify the calling application as part of etiquette.
    _APP_URL = "https://github.com/jyjulianwong/PolarRec"
    _APP_MAILTO = "jyw19@ic.ac.uk"

    # Max. # of results to be returned by each citation query made.
    _MAX_QUERY_RESULTS_RETD = hp.MAX_CIT_QUERY_RESULTS_RETD

    @classmethod
    def get_citation_count(cls, resources):
        """
        :param resources: The target resources.
        :type resources: list[Resource]
        :return: The number of times the resources had been cited previously.
        :rtype: dict[Resource, int]
        """
        pass

    @classmethod
    def get_references(cls, resources):
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
