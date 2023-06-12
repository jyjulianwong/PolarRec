"""
Resource database adapter for the IEEE Xplore library.
"""
import os
import random
import requests
from config import Config
from models.custom_logger import log
from models.persistent_cache import PersistentCache
from models.resource import Resource
from models.resource_database_adapters.query_builder import QueryBuilder


class IEEEXploreQueryBuilder(QueryBuilder):
    # To minimise the number of API calls made, cache the results.
    # This is used for development purposes only,
    # as IEEE Xplore has a limit on how many API calls can be made per day.
    _REQUEST_DATA_CACHE_FILEPATH = os.path.join(
        Config.APP_ROOT_DIR,
        "ieee-xplore-query-builder-cache.json"
    )
    # Create a variable to store the cached data that is in the JSON file.
    _request_data_cache = {}

    @classmethod
    def get_id(cls):
        return "ieeexplore"

    def __init__(self):
        super().__init__()
        self._API_KEY = Config.IEEE_XPLORE_API_KEY
        self._API_URL_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        self._query_args = {}
        self._authors = []
        self._title = ""
        self._keywords = []

        # Only load the cached data from the JSON file if in development env.
        is_dev_env = os.environ.get("FLASK_ENV", "development") == "development"
        if is_dev_env and PersistentCache.cache_enabled():
            self._request_data_cache = PersistentCache.load_cache_file(
                self._REQUEST_DATA_CACHE_FILEPATH
            )

    def _get_request_data(self, query_args):
        """
        :param query_args: The query parameters passed on to the API.
        :type query_args: dict[str, str]
        :return: The JSON data returned by the API request.
        :rtype: None | dict
        """
        try:
            # Create a Session object to store all the current API request.
            req_session = requests.Session()
            req = requests.Request(
                "GET",
                self._API_URL_BASE,
                params=query_args,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            # Prepare the request to preview the generated URL.
            req_prepped = req_session.prepare_request(req)

            if req_prepped.url in self._request_data_cache:
                # This request has been previously stored in cache.
                return self._request_data_cache[req_prepped.url]

            # Send a new request to the API.
            res = req_session.send(req_prepped, timeout=10)
        except Exception as err:
            log(str(err), "IEEEXploreQueryBuilder", "error")
            return None

        # Detect any errors or empty responses.
        if res.status_code != 200:
            # IEEE Xplore has a limit on how many API calls can be made per day.
            log(f"Got {res} from {res.url}", "IEEEXploreQueryBuilder", "error")
            return None

        log(f"Successful response from {res.url}", "IEEEXploreQueryBuilder")

        # Save the response of this new request in cache.
        self._request_data_cache[res.url] = res.json()
        return res.json()

    def _get_candidate_resources(self, query_args):
        """
        :param query_args: The query parameters passed on to the API.
        :type query_args: dict[str, str]
        :return: The candidate resources that are to be filtered later on.
        :rtype: list[Resource]
        """
        res = self._get_request_data(query_args)
        if res is None:
            # This should not happen.
            return []
        if res["total_records"] == 0:
            # No results were returned for this particular query.
            return []

        # Transform response data into Resource objects.
        resources: list[Resource] = []
        for resource_data in res["articles"]:
            author_data_list = resource_data["authors"]["authors"]
            authors = [data["full_name"] for data in author_data_list]

            # Set compulsory fields.
            resource_args = {
                "authors": authors,
                "title": resource_data["title"],
                "year": resource_data["publication_year"],
                "url": resource_data["html_url"]
            }

            # Set optional fields that are not necessarily returned every time.
            if "content_type" in resource_data:
                if resource_data["content_type"] == "Conferences":
                    if "publication_title" in resource_data:
                        resource_args["conference_name"] = resource_data[
                            "publication_title"
                        ]
            if "conference_location" in resource_data:
                resource_args["conference_location"] = resource_data[
                    "conference_location"
                ]
            if "abstract" in resource_data:
                resource_args["abstract"] = resource_data["abstract"]
            if "doi" in resource_data:
                resource_args["doi"] = resource_data["doi"]

            # Add the result as a Resource object.
            resources.append(Resource(resource_args))

        return resources

    def set_authors(self, authors):
        """
        IEEE Xplore has a limit on how many API calls can be made per day.
        One API call needs to be made per author because the API currently does
        not support queries for multiple authors at once.
        Therefore, the maximum number of authors included in the query has to be
        capped.
        """
        max_author_count = 3
        self._authors = authors[:min(len(authors), max_author_count)]

    def set_title(self, title):
        self._title = title

    def set_min_date(self, min_date):
        self._query_args["min_date"] = min_date

    def set_max_date(self, max_date):
        self._query_args["max_date"] = max_date

    def add_keyword(self, keyword):
        if " " in keyword:
            keyword = f'"{keyword}"'
        self._keywords.append(keyword)

    def get_resources(
        self,
        max_resources_returned,
        must_have_all_fields=True,
        summarise_results_data=False
    ):
        # Process query parameters.
        if len(self._keywords) > 0:
            bool_op = " AND " if must_have_all_fields else " OR "
            self._query_args["querytext"] = bool_op.join(self._keywords)
            self._query_args["querytext"] = f"({self._query_args['querytext']})"
        if len(self._title) > 0:
            self._query_args["article_title"] = self._title
        # The minimum result return length for IEEE Xplore API is 2.
        self._query_args["max_records"] = max(max_resources_returned, 2)
        self._query_args["apikey"] = self._API_KEY

        ress: list[Resource] = []

        # Additional engineering to handle queries with multiple authors.
        if len(self._authors) == 0:
            # The query does not contain authors.
            ress = self._get_candidate_resources(self._query_args)

        if len(self._authors) == 1:
            # The query contains 1 author.
            formatted_name = self._get_joined_author_name(self._authors[0], "+")
            self._query_args["author"] = formatted_name
            ress = self._get_candidate_resources(self._query_args)

        if len(self._authors) >= 2:
            # The query contains multiple authors.
            # Collect candidate resources that are to be filtered later on.
            cand_ress: list[Resource] = []
            for author in self._authors:
                formatted_name = self._get_joined_author_name(author, "+")
                self._query_args["author"] = formatted_name
                # Collect redundantly more candidates than necessary,
                # and increase the odds of finding the desired resources.
                self._query_args["max_records"] = 100
                cand_ress += self._get_candidate_resources(self._query_args)

            if not must_have_all_fields:
                # No need to remove candidates if only some authors are required.
                # Shuffle the candidates. Otherwise,
                # the first few authors will occupy the whole list.
                random.shuffle(cand_ress)
                ress = cand_ress[:min(len(cand_ress), max_resources_returned)]
            else:
                # Remove candidates that do not contain all the required authors.
                for resource in cand_ress:
                    # NoneType check.
                    if resource.authors is None:
                        continue

                    # Keep track of whether a required author couldn't be found.
                    remove = False
                    resource_last_names = [
                        self._get_author_last_name(a) for a in resource.authors
                    ]
                    for author in self._authors:
                        required_last_name = self._get_author_last_name(author)
                        # Only compare the last names of two authors.
                        # First names can be abbreviated, causing false rejects.
                        if required_last_name not in resource_last_names:
                            # A required author couldn't be found in this resource.
                            remove = True
                    if not remove:
                        # All the required authors were found.
                        ress.append(resource)

                ress = ress[:min(len(ress), max_resources_returned)]

        if summarise_results_data:
            self._summarise_results_data(ress)

        # Only save the data to cache if in development env.
        is_dev_env = os.environ.get("FLASK_ENV", "development") == "development"
        if is_dev_env and PersistentCache.cache_enabled():
            PersistentCache.save_cache_file(
                self._REQUEST_DATA_CACHE_FILEPATH, data=self._request_data_cache
            )

        return ress


if __name__ == '__main__':
    print("\nIEEEXploreQueryBuilder: Execute a query with one author")

    sample_query_builder = IEEEXploreQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    sample_query_builder.add_keyword("deep learning")
    sample_query_builder.set_authors(["Vijay Badrinarayanan"])
    resources = sample_query_builder.get_resources(
        20,
        must_have_all_fields=False
    )
    for i, resource in enumerate(resources):
        print(f"IEEEXploreQueryBuilder: get_resources: [{i}]:")
        print(f"\t{resource.title}")
        print(f"\t{resource.authors}")
        print(f"\t{resource.doi}")
        print(f"\t{resource.url}")
    print(f"IEEEXploreQueryBuilder: get_resources: len: {len(resources)}")

    print("\nIEEEXploreQueryBuilder: Execute a query with multiple authors")

    sample_query_builder = IEEEXploreQueryBuilder()
    sample_query_builder.set_authors(["Kai Gao", "Carly Donahue"])
    resources = sample_query_builder.get_resources(10)
    for i, resource in enumerate(resources):
        print(f"IEEEXploreQueryBuilder: get_resources: [{i}]:")
        print(f"\t{resource.title}")
        print(f"\t{resource.authors}")
        print(f"\t{resource.doi}")
        print(f"\t{resource.url}")
    print(f"IEEEXploreQueryBuilder: get_resources: len: {len(resources)}")

    # Analyse the fields returned by a typical query.
    sample_query_builder = IEEEXploreQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    resources = sample_query_builder.get_resources(
        50,
        summarise_results_data=True
    )
    print(f"IEEEXploreQueryBuilder: get_resources: len: {len(resources)}")
