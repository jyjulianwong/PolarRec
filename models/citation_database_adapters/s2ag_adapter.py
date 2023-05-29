"""
Citation database adapter for the Semantic Scholar Academic Graph (S2AG) API.
"""
import asyncio
import aiohttp
import json
import os
import requests
import time
from config import Config
from models.persistent_cache import PersistentCache
from models.citation_database_adapters.adapter import Adapter
from models.custom_logger import log, log_extended_line
from models.resource import Resource


class S2agAdapter(Adapter):
    # Users with API keys can send up to 100 requests per second.
    _API_KEY = Config.S2_API_KEY
    _API_URL_SINGLE_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
    _API_URL_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
    _API_RETURNED_RESOURCE_FIELDS = "paperId,title,citationCount,influentialCitationCount,references"
    # The max. number of requests to send in parallel in a single batch.
    # Users with API keys can send up to 100 requests per second.
    # Stay well below this limit for redundancy to prevent blocked requests.
    _API_REQ_BATCH_SIZE = 25

    # To minimise the number of API calls per resource, cache the results.
    # This is recommended for etiquette purposes in the documentation.
    _REQUEST_DATA_CACHE_FILEPATH = os.path.join(
        Config.APP_ROOT_DIR,
        "s2ag-adapter-cache.json"
    )
    _request_data_cache = {}

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
        clean_title = Resource.get_comparable_str(resource.title)
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
    def _get_req_url_str(cls, resource):
        """
        :param resource: The target resource.
        :type resource: Resource
        :return: The full request URL string.
        :rtype: str
        """
        param_str = cls._get_query_str(resource)
        param_str += f"&fields={cls._API_RETURNED_RESOURCE_FIELDS}"
        param_str += f"&offset=0"
        param_str += f"&limit={cls._MAX_QUERY_RESULTS_RETD}"
        return cls._API_URL_SINGLE_BASE + param_str

    @classmethod
    def _get_req_headers(cls):
        return {
            "User-Agent": f"PolarRec ({cls._APP_URL}; mailto:{cls._APP_MAILTO})",
            "Content-Type": "application/json",
            "x-api-key": cls._API_KEY
        }

    @classmethod
    def _add_req_data_cache_entry(cls, resource, data):
        cls._request_data_cache[resource.title] = data

    @classmethod
    async def _get_req_data_async_task(cls, resource, session):
        """
        :param resource: The target resources.
        :type resource: Resource
        :param session: The AIOHTTP client session.
        :type session: aiohttp.client.ClientSession
        :return: The API response data for this resource.
        :rtype: None | dict
        """
        url = cls._get_req_url_str(resource)
        headers = cls._get_req_headers()
        try:
            async with session.get(
                url=url,
                headers=headers,
                timeout=10
            ) as response:
                res = await response.read()
                res = json.loads(res.decode("utf-8"))
                log(f"Successful response from {url}", "S2agAdapter")

                if "total" not in res or res["total"] == 0:
                    # When the target resource cannot be found.
                    return None

                for cand_data in res["data"]:
                    cand_title = Resource.get_comparable_str(cand_data["title"])
                    targ_title = Resource.get_comparable_str(resource.title)
                    if cand_title == targ_title:
                        # When the target resource has been found.
                        cls._add_req_data_cache_entry(resource, cand_data)
                        return cand_data
        except Exception as err:
            log(str(err), "S2agAdapter", "error")

        # When the target resource cannot be found.
        return None

    @classmethod
    async def _get_req_data_async_batch(cls, resources):
        """
        :param resources: The target resources.
        :type resources: list[Resource]
        :return: The API response data for each resource.
        :rtype: list[None | dict]
        """
        async with aiohttp.ClientSession() as session:
            res_list = await asyncio.gather(
                *[cls._get_req_data_async_task(r, session) for r in resources]
            )
        return res_list

    @classmethod
    def _get_req_data(cls, resources):
        """
        :param resources: The target resources.
        :type resources: list[Resource]
        :return: The JSON data returned by the API request.
        :rtype: dict[Resource, None | dict]
        """
        result = {}

        # Sort target resources into those that can be queried in batches,
        # and those that cannot.
        ress_with_dois: list[Resource] = []
        ress_with_no_dois: list[Resource] = []
        for resource in resources:
            if resource.title in cls._request_data_cache:
                result[resource] = cls._request_data_cache[resource.title]
            elif resource.doi is None:
                ress_with_no_dois.append(resource)
            else:
                ress_with_dois.append(resource)

        # Collect data for resources with DOIs in a single POST request.
        if len(ress_with_dois) > 0:
            try:
                res = requests.post(
                    cls._API_URL_BATCH,
                    params={"fields": cls._API_RETURNED_RESOURCE_FIELDS},
                    headers=cls._get_req_headers(),
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
                log_extended_line(f"Response.text: {res.text}")
                for resource in ress_with_dois:
                    result[resource] = None
            else:
                log(f"Successful response from {res.url}", "S2agAdapter")

                res = res.json()
                for i in range(len(ress_with_dois)):
                    cls._add_req_data_cache_entry(ress_with_dois[i], res[i])
                    result[ress_with_dois[i]] = res[i]

        # Divide list of resources with no DOIs into smaller chunks.
        ress_with_no_dois_chunks = [
            ress_with_no_dois[x:x + cls._API_REQ_BATCH_SIZE] for x in range(
                0, len(ress_with_no_dois), cls._API_REQ_BATCH_SIZE
            )
        ]
        # Collect data for resources with no DOIs in parallel for each chunk.
        for chunk in ress_with_no_dois_chunks:
            res_list = asyncio.run(cls._get_req_data_async_batch(chunk))
            for i, resource in enumerate(chunk):
                # Response data is returned in the same order as the input list.
                result[resource] = res_list[i]

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
    def get_citation_count(cls, resources):
        if PersistentCache.cache_enabled():
            cls._request_data_cache = PersistentCache.load_cache_file(
                cls._REQUEST_DATA_CACHE_FILEPATH
            )

        data_dict = cls._get_req_data(resources)

        cit_count_dict: dict[Resource, int] = {}
        for resource, data in data_dict.items():
            if data is None:
                cit_count_dict[resource] = -1
            else:
                cit_count_dict[resource] = data["citationCount"]

        if PersistentCache.cache_enabled():
            PersistentCache.save_cache_file(
                cls._REQUEST_DATA_CACHE_FILEPATH, data=cls._request_data_cache
            )

        return cit_count_dict

    @classmethod
    def get_references(cls, resources):
        if PersistentCache.cache_enabled():
            cls._request_data_cache = PersistentCache.load_cache_file(
                cls._REQUEST_DATA_CACHE_FILEPATH
            )

        data_dict = cls._get_req_data(resources)

        ref_list_dict: dict[Resource, list[Resource]] = {}
        for resource, data in data_dict.items():
            ref_list_dict[resource] = cls._get_req_ref_data_as_ress(data)

        if PersistentCache.cache_enabled():
            PersistentCache.save_cache_file(
                cls._REQUEST_DATA_CACHE_FILEPATH, data=cls._request_data_cache
            )

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

        request_url_str = S2agAdapter._get_req_url_str(target_resource)
        print(f"S2agAdapter: request_url_str: {request_url_str}")

        t1 = time.time()
        citation_count = S2agAdapter.get_citation_count([target_resource])
        citation_count = citation_count[target_resource]
        print(f"S2agAdapter: citation_count: {citation_count}")
        t2 = time.time()
        print(f"S2agAdapter: Time taken to execute: {t2 - t1} seconds")

        print("\nS2agAdapter: Collect reference list for a single resource")

        t1 = time.time()
        references = S2agAdapter.get_references([target_resource])
        references = references[target_resource]
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
    ref_list_dict = S2agAdapter.get_references(
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
