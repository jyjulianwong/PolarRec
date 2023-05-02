"""
Objects and methods for ranking academic resources based on citations.
This module uses the Semantic Scholar Academic Graph (S2AG) API.
"""
import requests
import string
import time
from models.citation_database_adapters.adapter import Adapter
from models.resource import Resource


class S2agAdapter(Adapter):
    API_URL_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    # To minimise the number of API calls per resource, cache the results.
    # This is recommended for etiquette purposes in the documentation.
    request_data_cache = {}

    @classmethod
    def _get_query_str(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The query component of the request URL string.
        :rtype: str
        """
        # Use the resource's title and search for it.
        # Remove punctuation from title string.
        clean_title = resource.title.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        title_words = clean_title.split()
        # Remove any whitespace to avoid unnecessary "+"s in the query string.
        title_words = [word for word in title_words if word != ""]
        title_query_str = "+".join(title_words)

        query_str = f"?query={title_query_str}"

        if resource.year is not None:
            # Use the resource's year to eliminate unwanted search results.
            query_str += f"&year={resource.year}"

        return query_str

    @classmethod
    def _get_request_url_str(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The full request URL string.
        :rtype: str
        """
        param_str = cls._get_query_str(resource)
        param_str += "&fields=paperId,title,citationCount,influentialCitationCount,references"
        param_str += f"&limit={cls.MAX_SEARCH_RESULTS}"
        return cls.API_URL_BASE + param_str

    @classmethod
    def _add_request_data_cache_entry(cls, resource, data):
        cls.request_data_cache[resource.title] = data

    @classmethod
    def _get_request_data(cls, resource):
        if resource.title in cls.request_data_cache:
            return cls.request_data_cache[resource.title]

        headers = {
            "User-Agent": f"PolarRec ({cls.APP_URL}; mailto:{cls.APP_MAILTO})",
            "Content-Type": "application/json"
        }
        try:
            res = requests.get(
                cls._get_request_url_str(resource),
                headers=headers,
                timeout=10
            ).json()
        except Exception as err:
            print(f"S2agAdapter: {err}")
            return None

        if "total" not in res or res["total"] == 0:
            # When the target resource cannot be found.
            cls._add_request_data_cache_entry(resource, None)
            return None

        for cand_data in res["data"]:
            if cand_data["title"] == resource.title:
                # When the target resource has been found.
                cls._add_request_data_cache_entry(resource, cand_data)
                return cand_data

        cls._add_request_data_cache_entry(resource, None)
        return None

    @classmethod
    def get_citation_count(cls, resource):
        data = cls._get_request_data(resource)
        if data is None:
            return -1
        return data["citationCount"]

    @classmethod
    def get_references(cls, resource):
        data = cls._get_request_data(resource)
        if data is None:
            return []

        references: list[Resource] = []
        for reference_data in data["references"]:
            # TODO: Data typically only contains paperId and title.
            resource_args = {}
            if "authors" in reference_data:
                resource_args["authors"] = []
                for author_data in reference_data["authors"]:
                    resource_args["authors"].append(author_data["name"])
            if "title" in reference_data:
                resource_args["title"] = reference_data["title"]
            if "year" in reference_data:
                resource_args["year"] = reference_data["year"]
            if "url" in reference_data:
                resource_args["url"] = reference_data["url"]
            references.append(Resource(resource_args))
        return references


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
        target_resource = Resource(target_data)

        request_url_str = S2agAdapter._get_request_url_str(target_resource)
        print(f"citation_counts: request_url_str: {request_url_str}")

        t1 = time.time()
        citation_count = S2agAdapter.get_citation_count(target_resource)
        print(f"citation_counts: citation_count: {citation_count}")
        t2 = time.time()
        print(f"citation_counts: Time taken to execute: {t2 - t1} seconds")

        t1 = time.time()
        references = S2agAdapter.get_references(target_resource)
        print(f"citation_counts: len(references): {len(references)}")
        t2 = time.time()
        print(f"citation_counts: Time taken to execute: {t2 - t1} seconds")
