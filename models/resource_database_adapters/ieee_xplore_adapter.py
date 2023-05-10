"""
Resource database adapter for the IEEE Xplore library.
"""
import requests
from config import Config
from models.resource import Resource
from models.resource_database_adapters.adapter import QueryBuilder


class IEEEXploreQueryBuilder(QueryBuilder):
    def __init__(self):
        super().__init__()
        self._API_KEY = Config.IEEE_XPLORE_API_KEY
        self._API_URL_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        self._query_args = {}
        self._keywords = []

    def set_authors(self, authors):
        # TODO: IEEE Xplore does not support queries involving multiple authors.
        self._query_args["author"] = authors[0]

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
        if len(self._keywords) > 0:
            bool_op = " AND " if must_have_all_fields else " OR "
            self._query_args["querytext"] = bool_op.join(self._keywords)
            self._query_args["querytext"] = f"({self._query_args['querytext']})"
        self._query_args["max_records"] = max_resources_returned
        self._query_args["apikey"] = self._API_KEY
        try:
            res = requests.get(
                self._API_URL_BASE,
                params=self._query_args,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
        except Exception as err:
            print(f"IEEEXploreQueryBuilder: {err}")
            return []

        if res.status_code != 200:
            # IEEE Xplore has a limit on how many API calls can be made per day.
            print(f"IEEEXploreQueryBuilder: {res}: {res.url}")
            return []

        res = res.json()
        if res["total_records"] == 0:
            return []

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


if __name__ == '__main__':
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
