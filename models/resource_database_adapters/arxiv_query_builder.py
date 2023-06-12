"""
Resource database adapter for the arXiv library.
"""
import urllib
import xmltodict
from models.custom_logger import log
from models.resource import Resource
from models.resource_database_adapters.query_builder import QueryBuilder


class ArxivQueryBuilder(QueryBuilder):
    @classmethod
    def get_id(cls):
        return "arxiv"

    def __init__(self):
        super().__init__()
        self._API_URL_BASE = "http://export.arxiv.org/api/query"
        self._authors = None
        self._title = None
        self._min_date = None
        self._max_date = None
        self._keywords = []

    def _get_request_data(self, query_args):
        """
        :param query_args: The query parameters passed on to the API.
        :type query_args: dict[str, str]
        :return: The JSON data returned by the API request.
        :rtype: None | dict
        """
        # Generate the URL string with query parameters.
        url = self._API_URL_BASE + "?" + "&".join(
            [f"{key}={val}" for key, val in query_args.items()]
        )
        # Replace any special characters and make the string URL-friendly.
        url = self._get_translated_url_str(url)
        try:
            # Send a HTTP request to the API.
            res = urllib.request.urlopen(url)
            # Decode the response from bytes to string data.
            res = res.read().decode("utf-8")
            # Decode the response from XML to JSON-like format.
            res = xmltodict.parse(res)
        except Exception as err:
            log(str(err), "ArxivQueryBuilder", "error")
            return None

        log(f"Successful response from {url}", "ArxivQueryBuilder")

        return res

    def set_authors(self, authors):
        self._authors = authors

    def set_title(self, title):
        self._title = title

    def set_min_date(self, min_date):
        self._min_date = min_date

    def set_max_date(self, max_date):
        self._max_date = max_date

    def add_keyword(self, keyword):
        self._keywords.append(keyword)

    def get_resources(
        self,
        max_resources_returned,
        must_have_all_fields=True,
        summarise_results_data=False
    ):
        # The "search_query" query parameters to be passed onto the API.
        search_query = []

        if self._authors:
            for author in self._authors:
                formatted_name = self._get_joined_author_name(author, "_", True)
                search_query.append(f"au:{formatted_name}")

        if self._title:
            search_query.append(f"ti:{self._title.replace(' ', '+').lower()}")

        if self._min_date or self._max_date:
            # FIXME: Date-based search queries are not supported by arXiv.
            pass

        if self._keywords:
            for keyword in self._keywords:
                try:
                    # Only allow ASCII characters in the URL.
                    keyword.encode("ascii")
                except UnicodeEncodeError:
                    continue

                search_query.append(
                    f"all:%22{keyword.replace(' ', '+').lower()}%22"
                )

        # Generate the URL query parameters as a dict.
        query_op = "AND" if must_have_all_fields else "OR"
        query_args = {
            "search_query": f"+{query_op}+".join(search_query),
            "start": 0,
            "max_results": max_resources_returned
        }

        # Send the API request and decode the response.
        res = self._get_request_data(query_args)
        if res is None:
            # This should not happen.
            return []
        if res["feed"]["opensearch:totalResults"]["#text"] == "0":
            # No results were returned for this particular query.
            return []

        # Extract the list of results returned.
        resource_data_list = res["feed"]["entry"]
        if not isinstance(res["feed"]["entry"], list):
            # xmltodict changes the return type if there is only one entry.
            resource_data_list = [res["feed"]["entry"]]

        resources: list[Resource] = []
        for resource_data in resource_data_list:
            if not isinstance(resource_data, dict):
                log(
                    f"Expected type 'dict' from result returned, got '{type(resource_data)}' instead: {resource_data}",
                    "ArxivQueryBuilder",
                    "error"
                )
                continue

            if isinstance(resource_data["author"], list):
                authors = [data["name"] for data in resource_data["author"]]
            else:
                # xmltodict changes the return type if there is only one author.
                # We have to create a singleton list ourselves.
                authors = [resource_data["author"]["name"]]

            # Extract the year and month components from the date.
            date_components = resource_data["published"].split("-")

            # Set compulsory fields.
            resource_args = {
                "authors": authors,
                "title": resource_data["title"].replace("\n ", ""),
                "year": int(date_components[0]),
                "month": int(date_components[1]),
                "abstract": resource_data["summary"],
                "url": resource_data["id"]
            }

            # Set optional fields that are not necessarily returned every time.
            if "doi" in resource_data:
                resource_args["doi"] = resource_data["doi"]
            elif "arxiv:doi" in resource_data:
                resource_args["doi"] = resource_data["arxiv:doi"]["#text"]

            # Add the result as a Resource object.
            resources.append(Resource(resource_args))

        if summarise_results_data:
            self._summarise_results_data(resources)

        return resources


if __name__ == '__main__':
    sample_query_builder = ArxivQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    sample_query_builder.add_keyword("deep learning")
    sample_query_builder.set_authors(["Vijay Badrinarayanan"])
    resources = sample_query_builder.get_resources(20)
    for i, resource in enumerate(resources):
        print(f"ArxivQueryBuilder: get_resources: [{i}]:")
        print(f"\t{resource.title}")
        print(f"\t{resource.authors}")
        print(f"\t{resource.doi}")
        print(f"\t{resource.url}")
    print(f"ArxivQueryBuilder: get_resources: len: {len(resources)}")

    # Analyse the fields returned by a typical query.
    sample_query_builder = ArxivQueryBuilder()
    sample_query_builder.add_keyword("convolutional")
    resources = sample_query_builder.get_resources(
        50,
        summarise_results_data=True
    )
    print(f"ArxivQueryBuilder: get_resources: len: {len(resources)}")
