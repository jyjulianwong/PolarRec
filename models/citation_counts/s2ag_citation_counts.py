"""
Objects and methods for ranking academic resources based on citation counts.
This module uses the Semantic Scholar Academic Graph (S2AG) API.
"""
import requests
import string
import time
from models.resource import Resource

API_URL_BASE = "https://api.semanticscholar.org/graph/v1/paper"
API_URL_SEARCH_BASE = f"{API_URL_BASE}/search"
API_URL_MAILTO = "jyw19@ic.ac.uk"
MAX_SEARCH_RESULTS = 10


def _get_query_str(resource):
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


def _get_request_url_str(resource):
    """
    :param resource: The target resource.
    :type resource: Resource
    :return: The full request URL string.
    :rtype: str
    """
    param_str = "&fields=paperId,title,citationCount,influentialCitationCount"
    param_str += f"&limit={MAX_SEARCH_RESULTS}"
    return API_URL_SEARCH_BASE + _get_query_str(resource) + param_str


def get_citation_count(resource):
    """
    :param resource: The target resource.
    :type resource: Resource
    :return: The citation count quoted by Semantic Scholar.
    :rtype: int
    """
    headers = {
        "User-Agent": f"PolarRec (https://github.com/jyjulianwong/PolarRec; mailto:{API_URL_MAILTO})",
        "Content-Type": "application/json"
    }
    try:
        res = requests.get(
            _get_request_url_str(resource),
            headers=headers,
            timeout=10
        ).json()
    except:
        return -1

    if "total" not in res or res["total"] == 0:
        # When the target resource cannot be found.
        return -1

    for cand_data in res["data"]:
        if cand_data["title"] == resource.title:
            # When the target resource has been found.
            return cand_data["citationCount"]

    return -1


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

        request_url_str = _get_request_url_str(target_resource)
        print(f"citation_counts: request_url_str: {request_url_str}")

        t1 = time.time()
        citation_count = get_citation_count(target_resource)
        print(f"citation_counts: citation_count: {citation_count}")
        t2 = time.time()
        print(f"citation_counts: Time taken to execute: {t2 - t1} seconds")
