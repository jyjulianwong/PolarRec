"""
Citation database adapter for the Semantic Scholar Academic Graph (S2AG) API.
"""
import requests
import string
import time
from models.citation_database_adapters.adapter import Adapter
from models.custom_logger import log
from models.resource import Resource


class S2agAdapter(Adapter):
    API_URL_SINGLE_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
    API_URL_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
    API_RETURNED_FIELDS = "paperId,title,citationCount,influentialCitationCount,references"

    # To minimise the number of API calls per resource, cache the results.
    # This only caches incoming data within a single API call, not across calls.
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
        param_str += f"&fields={cls.API_RETURNED_FIELDS}"
        param_str += f"&limit={cls.MAX_SEARCH_RESULTS}"
        return cls.API_URL_SINGLE_BASE + param_str

    @classmethod
    def _add_request_data_cache_entry(cls, resource, data):
        cls.request_data_cache[resource.title] = data

    @classmethod
    def _get_request_data(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The JSON data returned by the API request.
        :rtype: None | dict
        """
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
            )
        except Exception as err:
            log(str(err), "S2agAdapter", "error")
            return None

        # Detect any errors or empty responses.
        if res.status_code != 200:
            # Non-partners can only send 5,000 requests per 5 minutes.
            log(f"Got {res} from {res.url}", "S2agAdapter", "error")
            return None

        res = res.json()
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
    def _get_req_data_in_batches(cls, resources):
        """
        :param resource: The target resources.
        :type resource: list[Resource]
        :return: The JSON data returned by the API request.
        :rtype: dict[Resource, None | dict]
        """
        # Sort target resources into those that can be queried in batches,
        # and those that cannot.
        ress_with_dois: list[Resource] = []
        ress_with_no_dois: list[Resource] = []
        for resource in resources:
            if resource.doi is None:
                ress_with_no_dois.append(resource)
            else:
                ress_with_dois.append(resource)

        result = {}

        try:
            res = requests.post(
                cls.API_URL_BATCH,
                params={"fields": cls.API_RETURNED_FIELDS},
                json={"ids": [res.doi for res in ress_with_dois]}
            )
        except Exception as err:
            log(str(err), "S2agAdapter", "error")
            for resource in ress_with_dois:
                result[resource] = None

        # Detect any errors or empty responses.
        if res.status_code != 200:
            # Non-partners can only send 5,000 requests per 5 minutes.
            log(f"Got {res} from {res.url}", "S2agAdapter", "error")
            for resource in ress_with_dois:
                result[resource] = None

        res = res.json()
        for i in range(len(ress_with_dois)):
            cls._add_request_data_cache_entry(ress_with_dois[i], res[i])
            result[ress_with_dois[i]] = res[i]

        for resource in ress_with_no_dois:
            data = cls._get_request_data(resource)
            result[resource] = data

        return result

    @classmethod
    def _get_req_ref_data_as_ress(cls, data):
        """
        :param data: The raw data returned from a request about a resource.
        :type data: dict
        :return: The list of references extracted from the raw data.
        :rtype: list[Resource]
        """
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

    @classmethod
    def get_citation_count(cls, resource):
        data = cls._get_request_data(resource)
        if data is None:
            return -1
        return data["citationCount"]

    @classmethod
    def get_references(cls, resource):
        data = cls._get_request_data(resource)
        return cls._get_req_ref_data_as_ress(data)

    @classmethod
    def get_references_in_batches(cls, resources):
        data_dict = cls._get_req_data_in_batches(resources)

        ref_list_dict: dict[Resource, list[Resource]] = {}
        for resource, data in data_dict.items():
            ref_list_dict[resource] = cls._get_req_ref_data_as_ress(data)

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
        print("\nS2agAdapter: Collect citation count for a single resource")

        target_resource = Resource(target_data)

        request_url_str = S2agAdapter._get_request_url_str(target_resource)
        print(f"S2agAdapter: request_url_str: {request_url_str}")

        t1 = time.time()
        citation_count = S2agAdapter.get_citation_count(target_resource)
        print(f"S2agAdapter: citation_count: {citation_count}")
        t2 = time.time()
        print(f"S2agAdapter: Time taken to execute: {t2 - t1} seconds")

        print("\nS2agAdapter: Collect reference list for a single resource")

        t1 = time.time()
        references = S2agAdapter.get_references(target_resource)
        print(f"S2agAdapter: len(references): {len(references)}")
        t2 = time.time()
        print(f"S2agAdapter: Time taken to execute: {t2 - t1} seconds")
        print(f"\tFirst 3 references of {target_resource}:")
        for reference in references[:min(len(references), 3)]:
            print(f"\t\t{reference.title}")
            print(f"\t\t\t{reference.authors}")
            print(f"\t\t\t{reference.year}")

    print("\nS2agAdapter: Collect reference lists for a batch of resources")

    t1 = time.time()
    ref_list_dict = S2agAdapter.get_references_in_batches(
        [Resource(target_data1), Resource(target_data2)]
    )
    print(f"S2agAdapter: len(ref_list_dict): {len(ref_list_dict)}")
    t2 = time.time()
    print(f"S2agAdapter: Time taken to execute: {t2 - t1} seconds")
    for resource, references in ref_list_dict.items():
        print(f"\t{resource}: {len(references)}")
        print(f"\t\tFirst 3 references of {resource}:")
        for reference in references[:min(len(references), 3)]:
            print(f"\t\t\t{reference.title}")
            print(f"\t\t\t\t{reference.authors}")
            print(f"\t\t\t\t{reference.year}")
