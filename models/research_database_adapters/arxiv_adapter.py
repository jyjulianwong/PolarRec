"""
Research database adapter for the arXiv library.
"""
import urllib
import xmltodict
from models.research_database_adapters.adapter import QueryBuilder
from models.resource import Resource


class ArxivQueryBuilder(QueryBuilder):
    def __init__(self):
        super().__init__()
        self._API_URL_BASE = "http://export.arxiv.org/api/query"
        self._author = None
        self._min_date = None
        self._max_date = None
        self._keywords = []

    def set_author(self, author):
        self._author = author

    def set_min_date(self, min_date):
        self._min_date = min_date

    def set_max_date(self, max_date):
        self._max_date = max_date

    def add_keyword(self, keyword):
        self._keywords.append(keyword)

    def get_resources(self):
        search_query = []
        if self._author:
            search_query.append(f"au:{self._author.replace(' ', '_').lower()}")
        if self._min_date or self._max_date:
            # TODO: Not supported by ArXiv API.
            pass
        if self._keywords:
            for keyword in self._keywords:
                search_query.append(
                    f"all:%22{keyword.replace(' ', '+').lower()}%22"
                )

        query_args = {
            "search_query": "+AND+".join(search_query),
            "start": 0,
            "max_results": self._MAX_RES_ITEMS_COUNT
        }

        url = self._API_URL_BASE + "?" + "&".join(
            [f"{key}={val}" for key, val in query_args.items()]
        )
        # TODO: Handle failure.
        res = urllib.request.urlopen(url)
        res = res.read().decode("utf-8")
        res = xmltodict.parse(res)

        if res["feed"]["opensearch:totalResults"]["#text"] == "0":
            return []

        resources = []
        for resource_data in res["feed"]["entry"]:
            resources.append(Resource(
                title=resource_data["title"].replace("\n ", ""),
                authors=[data["name"] for data in resource_data["author"]],
                date=resource_data["published"],
                abstract=resource_data["summary"],
                url=resource_data["id"]
            ))
        return resources


if __name__ == '__main__':
    # TODO: Test.
    sample_query_builder = ArxivQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    sample_query_builder.add_keyword("deep learning")
    sample_query_builder.set_author("Badrinarayanan")
    resources = sample_query_builder.get_resources()
    resources = [resource.title for resource in resources]
    print(f"ArxivQueryBuilder: get_resources: {resources}")
    print(f"ArxivQueryBuilder: get_resources: len: {len(resources)}")
