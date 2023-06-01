"""
Objects and methods used to calculate the keyword extraction accuracy of the
recommendation algorithm.
"""
import requests
from config import Config
from models.custom_logger import log
from models.persistent_cache import PersistentCache
from models.evaluation import sample_resources as sr
from models.resource import Resource
from models.resource_rankers.keyword_ranker import KeywordRanker

KEYWORD_CACHE_FILEPATH = "keyword-extraction-cache.json"


def _get_request_data(resource):
    """
    :param resource: The resource to collect pre-defined keywords for.
    :type resource: Resource
    :return: The JSON data returned by the API.
    :rtype: None | dict
    """
    try:
        res = requests.get(
            "https://ieeexploreapi.ieee.org/api/v1/search/articles",
            params={
                "article_title": resource.title,
                # The minimum result return length for IEEE Xplore API is 2.
                "max_records": 2,
                "apikey": Config.IEEE_XPLORE_API_KEY,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=10
        )
    except Exception as err:
        log(str(err), "evaluation.keyword_extraction", "error")
        return None

    # Detect any errors or empty responses.
    if res.status_code != 200:
        # IEEE Xplore has a limit on how many API calls can be made per day.
        log(
            f"Got {res} from {res.url}",
            "evaluation.keyword_extraction",
            "error"
        )
        return None

    log(f"Successful response from {res.url}", "evaluation.keyword_extraction")

    return res.json()


def _get_predef_keywords(resource):
    """
    :param resource: The resource to collect pre-defined keywords for.
    :type resource: Resource
    :return: The mapping between resources and their pre-defined keywords.
    :rtype: None | list[str]
    """
    keyword_cache = PersistentCache.load_cache_file(KEYWORD_CACHE_FILEPATH)

    if resource.title in keyword_cache:
        return keyword_cache[resource.title]

    res = _get_request_data(resource)

    if res is None:
        return None
    if res["total_records"] == 0:
        return None

    resource_data = res["articles"][0]
    if "index_terms" not in resource_data:
        return None

    predef_keywords = []
    for kw_type, kw_data in resource_data["index_terms"].items():
        predef_keywords += kw_data["terms"]
    # Remove duplicates but preserve ordering.
    predef_keywords = list(dict.fromkeys(predef_keywords))
    keyword_cache[resource.title] = predef_keywords

    PersistentCache.save_cache_file(KEYWORD_CACHE_FILEPATH, keyword_cache)

    return predef_keywords


def get_kw_extraction_accuracy(target_resource):
    """
    :param target_resource: The target sample resource.
    :type target_resource: Resource
    :return: The keyword extraction accuracy of the extraction algorithm.
    :rtype: float
    """
    hit_count = 0
    miss_count = 0

    predef_keywords = _get_predef_keywords(target_resource)
    if predef_keywords is None:
        log(
            f"No pre-defined keywords found for '{target_resource.title}'",
            "evaluation.keyword_extraction",
            "error"
        )
        return 0.0
    # Only consider the top 20 pre-defined keywords.
    predef_keywords = predef_keywords[:min(len(predef_keywords), 20)]

    ranker_keywords = KeywordRanker.get_keywords(resources=[target_resource])
    # Only consider the top 20 keywords extracted by the algorithm.
    ranker_keywords = ranker_keywords[:min(len(ranker_keywords), 20)]

    for ranker_keyword in ranker_keywords:
        ranker_keyword = Resource.get_comparable_str(ranker_keyword)

        matched = False
        for predef_keyword in predef_keywords:
            predef_keyword = Resource.get_comparable_str(predef_keyword)

            common_words = list(set(ranker_keyword.split()).intersection(
                predef_keyword.split()
            ))

            if len(common_words) > len(predef_keyword.split()) // 2:
                matched = True
                hit_count += 1
                break

        if not matched:
            miss_count += 1

    return hit_count / (hit_count + miss_count)


if __name__ == "__main__":
    # Use the IEEE Xplore samples because we need pre-defined keyword data.
    sample_resources = sr.load_resources_from_json()[
        sr.IEEE_XPLORE_SAMPLE_FILEPATH
    ]

    keas: list[float] = []
    for i, resource in enumerate(sample_resources):
        kea = get_kw_extraction_accuracy(resource)
        keas.append(get_kw_extraction_accuracy(resource))
        print(f"Keyword extraction accuracy {i}: {kea}")

    mean_kea = sum(keas) / len(keas)
    print(f"Mean keyword extraction accuracy: {mean_kea}")
