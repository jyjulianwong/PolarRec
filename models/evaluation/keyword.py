"""
Objects and methods used to calculate the keyword accuracy of the
recommendation algorithm.
"""
import requests
import string
from config import Config
from models.custom_logger import log
from models.dev_cache import DevCache
from models.evaluation import sample_resources as sr
from models.hyperparams import Hyperparams as hp
from models.resource import Resource
from models.resource_rankers.keyword_ranker import KeywordRanker

KEYWORD_CACHE_FILEPATH = "keyword-cache.json"


def get_request_data(resource):
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
        log(str(err), "evaluation.keyword", "error")
        return None

    # Detect any errors or empty responses.
    if res.status_code != 200:
        # IEEE Xplore has a limit on how many API calls can be made per day.
        log(f"Got {res} from {res.url}", "evaluation.keyword", "error")
        return None

    log(f"Successful response from {res.url}", "evaluation.keyword")

    return res.json()


def get_resource_keyword_dict(resources):
    """
    :param resources: The resources to collect pre-defined keywords for.
    :type resources: list[Resource]
    :return: The mapping between resources and their pre-defined keywords.
    :rtype: dict[Resource, list[str]]
    """
    res_dw_dict: dict[Resource, list[str]] = {}

    keyword_cache = DevCache.load_cache_file(KEYWORD_CACHE_FILEPATH)

    for resource in resources:
        if resource.title in keyword_cache:
            res_dw_dict[resource] = keyword_cache[resource.title]
            continue

        res = get_request_data(resource)

        if res is None:
            res_dw_dict[resource] = []
            continue
        if res["total_records"] == 0:
            res_dw_dict[resource] = []
            continue

        resource_data = res["articles"][0]
        if "index_terms" in resource_data:
            predef_keywords = []
            for kw_type, kw_data in resource_data["index_terms"].items():
                predef_keywords += kw_data["terms"]
            # Remove duplicates but preserve ordering.
            predef_keywords = list(dict.fromkeys(predef_keywords))
            res_dw_dict[resource] = predef_keywords
            keyword_cache[resource.title] = predef_keywords
        else:
            res_dw_dict[resource] = []
            continue

    DevCache.save_cache_file(KEYWORD_CACHE_FILEPATH, keyword_cache)

    return res_dw_dict


def get_keyword_precision(target_resource):
    """
    :param target_resource: The target sample resource.
    :type target_resource: Resource
    :return: The keyword precision of the keyword extraction algorithm.
    :rtype: float
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    res_dw_dict = get_resource_keyword_dict(resources=[target_resource])
    predef_keywords = res_dw_dict[target_resource]
    ranker_keywords = KeywordRanker.get_keywords(resources=[target_resource])

    for ranker_keyword in ranker_keywords:
        ranker_keyword = "".join(
            " " if c in string.punctuation else c for c in ranker_keyword
        ).lower()

        matched = False
        for predef_keyword in predef_keywords:
            predef_keyword = "".join(
                " " if c in string.punctuation else c for c in predef_keyword
            ).lower()

            common_words = list(set(ranker_keyword.split(" ")).intersection(
                predef_keyword.split(" ")
            ))

            if len(common_words) > 0:
                matched = True
                true_positive += 1
                break

        if not matched:
            false_positive += 1

    return true_positive / (true_positive + false_positive)


if __name__ == "__main__":
    # TODO: Implement.
    pass
