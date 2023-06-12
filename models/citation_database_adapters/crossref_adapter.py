"""
Citation database adapter for the Crossref API.
This module is no longer being properly updated, as all citation database data
is now obtained through the S2agAdapter instead. This module now exists simply
as a demonstration of how the recommendation engine can support the use of
multiple Citation Database Adapters at the choice of the developer.
"""
import requests
import time
from models.citation_database_adapters.adapter import Adapter
from models.resource import Resource


class CrossrefAdapter(Adapter):
    API_URL_BASE = "https://api.crossref.org/works"

    @classmethod
    def _get_query_str(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The query component of the API request URL.
        :rtype: str
        """
        # Use the resource's title and search for it.
        # Remove punctuation from title string.
        clean_title = Resource.get_comparable_str(resource.title)
        title_words = clean_title.split()
        # Remove any whitespace to avoid unnecessary "+"s in the query string.
        title_words = [word for word in title_words if word != ""]
        title_query_str = "+".join(title_words)
        return f"?query.title={title_query_str}"

    @classmethod
    def _get_request_url_str(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The full API request URL with query parameters.
        :rtype: str
        """
        param_str = cls._get_query_str(resource)
        param_str += f"&rows={cls._MAX_QUERY_RESULTS_RETD}"
        param_str += f"&mailto={cls._APP_MAILTO}"
        return cls.API_URL_BASE + param_str

    @classmethod
    def get_citation_count(cls, resources):
        # FIXME: Not an efficient implementation.
        cit_count_dict: dict[Resource, int] = {}

        for resource in resources:
            headers = {
                "User-Agent": f"PolarRec ({cls._APP_URL}; mailto:{cls._APP_MAILTO})",
                "Content-Type": "application/json"
            }
            try:
                res = requests.get(
                    cls._get_request_url_str(resource),
                    headers=headers,
                    timeout=10
                ).json()
            except:
                cit_count_dict[resource] = -1

            if len(res["message"]["items"]) == 0:
                # When the target resource cannot be found.
                cit_count_dict[resource] = -1

            cit_count_dict[resource] = res["message"]["items"][0][
                "is-referenced-by-count"
            ]

        return cit_count_dict

    @classmethod
    def get_references(cls, resources):
        # FIXME: Reference data is not supported by the Crossref API.
        ref_list_dict = {}

        for resource in resources:
            ref_list_dict[resource] = []

        return ref_list_dict


if __name__ == "__main__":
    target_data1 = {
        "authors": ["Vijay Badrinarayanan", "Alex Kendall", "Roberto Cipolla"],
        "title": "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",
        "year": 2015,
        "doi": "10.1109/TPAMI.2016.2644615",
        "url": "https://ieeexplore.ieee.org/document/7803544"
    }
    target_data2 = {
        "authors": ["Does Not Exist"],
        "title": "Does Not Exist",
        "year": 1000,
        "doi": "Does Not Exist",
        "url": "Does Not Exist"
    }

    for target_data in [target_data1, target_data2]:
        print("\nCrossrefAdapter: Collect citation count for a single resource")

        target_resource = Resource(target_data)

        request_url_str = CrossrefAdapter._get_request_url_str(target_resource)
        print(f"CrossrefAdapter: request_url_str: {request_url_str}")

        t1 = time.time()
        citation_count = CrossrefAdapter.get_citation_count([target_resource])
        citation_count = citation_count[target_resource]
        print(f"CrossrefAdapter: citation_count: {citation_count}")
        t2 = time.time()
        print(f"CrossrefAdapter: Time taken to execute: {t2 - t1} seconds")

        print("\nCrossrefAdapter: Collect reference list for a single resource")

        t1 = time.time()
        references = CrossrefAdapter.get_references([target_resource])
        references = references[target_resource]
        print(f"CrossrefAdapter: len(references): {len(references)}")
        t2 = time.time()
        print(f"CrossrefAdapter: Time taken to execute: {t2 - t1} seconds")
