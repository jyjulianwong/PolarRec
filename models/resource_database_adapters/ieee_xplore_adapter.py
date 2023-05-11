"""
Resource database adapter for the IEEE Xplore library.
"""
import random
import requests
from config import Config
from models.custom_logger import log
from models.resource import Resource
from models.resource_database_adapters.adapter import QueryBuilder


class IEEEXploreQueryBuilder(QueryBuilder):
    def __init__(self):
        super().__init__()
        self._API_KEY = Config.IEEE_XPLORE_API_KEY
        self._API_URL_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        self._query_args = {}
        self._authors = []
        self._keywords = []

    def _get_candidate_resources(self, query_args):
        """
        :param query_args: The query parameters passed on to the API.
        :type query_args: dict[str, str]
        :return: The candidate resources that are to be filtered later on.
        :rtype: list[Resouce]
        """
        try:
            res = requests.get(
                self._API_URL_BASE,
                params=query_args,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
        except Exception as err:
            log(str(err), "IEEEXploreQueryBuilder", "error")
            return []

        # Detect any errors or empty responses.
        if res.status_code != 200:
            # IEEE Xplore has a limit on how many API calls can be made per day.
            log(f"Got {res} from {res.url}", "IEEEXploreQueryBuilder", "error")
            return []

        log(f"Successful response from {res.url}", "IEEEXploreQueryBuilder")
        res = res.json()
        if res["total_records"] == 0:
            return []

        # Transform response data into Resource objects.
        resources = []
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

            # Set optional fields.
            if "abstract" in resource_data:
                resource_args["abstract"] = resource_data["abstract"]
            if "doi" in resource_data:
                resource_args["doi"] = resource_data["doi"]

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
        must_have_all_fields=True
    ):
        # Process query parameters.
        if len(self._keywords) > 0:
            bool_op = " AND " if must_have_all_fields else " OR "
            self._query_args["querytext"] = bool_op.join(self._keywords)
            self._query_args["querytext"] = f"({self._query_args['querytext']})"
        self._query_args["max_records"] = max_resources_returned
        self._query_args["apikey"] = self._API_KEY

        # Collect candidate resources that are to be filtered later on.
        if len(self._authors) == 0:
            # The query does not contain authors.
            return self._get_candidate_resources(self._query_args)

        if len(self._authors) == 1:
            # The query contains 1 author.
            formatted_name = self._get_joined_author_name(self._authors[0], "+")
            self._query_args["author"] = formatted_name
            return self._get_candidate_resources(self._query_args)

        # The query contains multiple authors.
        cand_ress: list[Resource] = []
        for author in self._authors:
            formatted_name = self._get_joined_author_name(author, "+")
            self._query_args["author"] = formatted_name
            # Collect redundantly more candidates than necessary,
            # and increase the odds of finding the desired resources.
            self._query_args["max_records"] = 100
            cand_ress += self._get_candidate_resources(self._query_args)

        # No need to remove candidates if only some of the authors are required.
        if not must_have_all_fields:
            # Shuffle the candidates. Otherwise,
            # candidates of the first few authors will occupy the whole list.
            random.shuffle(cand_ress)
            return cand_ress[:min(len(cand_ress), max_resources_returned)]

        # Remove candidates that do not contain all the desired authors.
        resources: list[Resource] = []

        for resource in cand_ress:
            # NoneType check.
            if resource.authors is None:
                continue

            # Keep track of whether a required author could not be found.
            remove = False
            cand_last_names = [
                self._get_author_last_name(a) for a in resource.authors
            ]
            for author in self._authors:
                required_last_name = self._get_author_last_name(author)
                # Only compare the last names of two authors.
                # First names are sometimes abbreviated, causing false rejects.
                if required_last_name not in cand_last_names:
                    # A required author could not be found in this resource.
                    remove = True
            if not remove:
                # All the required authors were found.
                resources.append(resource)

        return resources[:min(len(resources), max_resources_returned)]


if __name__ == '__main__':
    # Test for queries involving one author.
    sample_query_builder = IEEEXploreQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    sample_query_builder.add_keyword("deep learning")
    sample_query_builder.set_authors(["Vijay Badrinarayanan"])
    resources = sample_query_builder.get_resources(
        10,
        must_have_all_fields=False
    )
    for i, resource in enumerate(resources):
        print(f"IEEEXploreQueryBuilder: get_resources: [{i}]:")
        print(f"\t{resource.title}")
        print(f"\t{resource.authors}")
        print(f"\t{resource.doi}")
        print(f"\t{resource.url}")
    print(f"IEEEXploreQueryBuilder: get_resources: len: {len(resources)}")

    # Test for queries involving multiple authors.
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
