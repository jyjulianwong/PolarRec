"""
Research database adapter for the IEEE Xplore library.
"""
import os
import requests
from flask import jsonify
from models.research_database_adapters.adapter import QueryBuilder

API_KEY = os.environ.get("IEEE_XPLORE_API_KEY")
API_URL_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


class IEEEXploreQueryBuilder(QueryBuilder):
    def __init__(self):
        super().__init__()
        self.query_args = {}
        self.keywords = []

    def set_author(self, author):
        self.query_args["author"] = author

    def set_start_date(self, start_date):
        self.query_args["start_date"] = start_date

    def set_end_date(self, end_date):
        self.query_args["end_date"] = end_date

    def add_keyword(self, keyword):
        self.keywords.append(keyword)

    def get_response(self):
        self.query_args["querytext"] = "%20AND%20".join(self.keywords)
        self.query_args["max_records"] = 200
        self.query_args["apikey"] = API_KEY
        res = requests.get(
            API_URL_BASE,
            params=self.query_args,
            timeout=10
        ).content
        # TODO: Handle failure.
        return jsonify(res)


if __name__ == '__main__':
    # TODO: Test.
    sample_query_builder = IEEEXploreQueryBuilder()
    sample_query_builder.add_keyword("computer")
    sample_query_builder.add_keyword("programming")
    sample_query_builder.set_author("Lisa C. Kaczmarczyk")
    res = sample_query_builder.get_response()
    print(f"IEEEXploreQueryBuilder: get_response: {res}")
