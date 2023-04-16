"""
Research database adapter for the IEEE Xplore library.
"""
import requests
from config import Config
from models.research_database_adapters.adapter import QueryBuilder


class IEEEXploreQueryBuilder(QueryBuilder):
    def __init__(self):
        super().__init__()
        self._API_KEY = Config.IEEE_XPLORE_API_KEY
        self._API_URL_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        self._query_args = {}
        self._keywords = []

    def set_authors(self, authors):
        # TODO: Support for multiple authors?
        self._query_args["author"] = ", ".join(authors)

    def set_min_date(self, min_date):
        self._query_args["min_date"] = min_date

    def set_max_date(self, max_date):
        self._query_args["max_date"] = max_date

    def add_keyword(self, keyword):
        self._keywords.append(keyword)

    def get_resources(
        self,
        max_resources_returned,
        must_have_all_fields=True
    ):
        self._query_args["querytext"] = "%20AND%20".join(self._keywords)
        self._query_args["max_records"] = max_resources_returned
        self._query_args["apikey"] = self._API_KEY
        res = requests.get(
            self._API_URL_BASE,
            params=self._query_args,
            timeout=10
        ).content
        # TODO: Handle failure.
        return res


if __name__ == '__main__':
    # TODO: Test.
    sample_query_builder = IEEEXploreQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    sample_query_builder.add_keyword("deep learning")
    sample_query_builder.set_authors(["Vijay Badrinarayanan"])
    resources = sample_query_builder.get_resources(10)
    resources = [resource.title for resource in resources]
    print(f"ArxivQueryBuilder: get_resources: {resources}")
    print(f"ArxivQueryBuilder: get_resources: len: {len(resources)}")
