"""
Objects and methods used to calculate the recommendation accuracy of the
recommendation algorithm.
"""
import requests
from config import Config
from models.custom_logger import log, log_extended_line
from models.persistent_cache import PersistentCache
from models.evaluation import sample_resources as sr
from models.recommend import get_recommended_resources
from models.resource import Resource
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker


class S2rAdapter:
    """
    Adapter for the Semantic Scholar Recommendations (S2R) API. To be used for
    evaluation and comparison purposes only, and not to be used within the main
    recommendation algorithm of this application.
    """
    # Used by APIs to identify the calling application as part of etiquette.
    _APP_URL = "https://github.com/jyjulianwong/PolarRec"
    _APP_MAILTO = "jyw19@ic.ac.uk"
    _API_KEY = Config.S2_API_KEY
    _API_URL_BASE = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"

    _REQUEST_DATA_CACHE_FILEPATH = "recommendation-cache.json"
    _request_data_cache = {}

    @classmethod
    def _get_request_url_str(cls, resource, max_results_retd):
        """
        :param resource: The target resource.
        :type resource: Resource
        :param max_results_retd: Max. number of results returned by the query.
        :type max_results_retd: int
        :return: The full request URL string.
        :rtype: str
        """
        param_str = f"fields=paperId,title,authors&limit={max_results_retd}"
        return f"{cls._API_URL_BASE}/{resource.doi}?{param_str}"

    @classmethod
    def _add_request_data_cache_entry(cls, resource, data):
        cls._request_data_cache[resource.title] = data

    @classmethod
    def _get_request_data(cls, resource, max_results_retd):
        """
        :param resource: The target resource.
        :type resource: Resource
        :param max_results_retd: Max. number of results returned by the query.
        :type max_results_retd: int
        :return: The JSON data returned by the API request.
        :rtype: None | list[dict]
        """
        if resource.title in cls._request_data_cache:
            return cls._request_data_cache[resource.title]

        headers = {
            "User-Agent": f"PolarRec ({cls._APP_URL}; mailto:{cls._APP_MAILTO})",
            "Content-Type": "application/json",
            "x-api-key": cls._API_KEY
        }
        try:
            res = requests.get(
                cls._get_request_url_str(resource, max_results_retd),
                headers=headers,
                timeout=10
            )
        except Exception as err:
            log(str(err), "S2rAdapter", "error")
            return []

        # Detect any errors or empty responses.
        if res.status_code != 200:
            # Non-partners can only send 5,000 requests per 5 minutes.
            log(f"Got {res} from {res.url}", "S2rAdapter", "error")
            log_extended_line(f"Response.text: {res.text}")
            return []

        log(f"Successful response from {res.url}", "S2rAdapter")

        res = res.json()
        cls._add_request_data_cache_entry(resource, res["recommendedPapers"])
        return res["recommendedPapers"]

    @classmethod
    def get_recommended_resources(cls, target_resource, max_results_retd):
        """
        :param target_resource: The target resource.
        :type target_resource: Resource
        :param max_results_retd: Max. number of results returned by the query.
        :type max_results_retd: int
        :return: The list of recommended resources.
        :rtype: list[Resource]
        """
        cls._request_data_cache = PersistentCache.load_cache_file(
            cls._REQUEST_DATA_CACHE_FILEPATH
        )

        res = cls._get_request_data(target_resource, max_results_retd)

        recommended_resources: list[Resource] = []
        for resource_data in res:
            resource_args = {
                "authors": [
                    a_data["name"] for a_data in resource_data["authors"]
                ],
                "title": resource_data["title"]
            }
            recommended_resources.append(Resource(args=resource_args))

        PersistentCache.save_cache_file(
            cls._REQUEST_DATA_CACHE_FILEPATH, data=cls._request_data_cache
        )

        return recommended_resources


def get_recommend_accuracy(target_resource, keyword_model):
    """
    :param target_resource: The target resource.
    :type target_resource: Resource
    :param keyword_model: The word embedding model to be used for keywords.
    :type keyword_model: Word2Vec.KeyedVectors
    :return: The recommendation accuracy of the recommendation algorithm.
    :rtype: float
    """
    max_compared_results = 30

    hit_count = 0
    miss_count = 0

    y_true = S2rAdapter.get_recommended_resources(
        target_resource=target_resource,
        max_results_retd=max_compared_results
    )
    _, y_pred = get_recommended_resources(
        target_resources=[target_resource],
        existing_resources=[],
        resource_filter=ResourceFilter({}),
        resource_database_ids=[],
        keyword_model=keyword_model
    )
    # Only consider the top resources returned by our algorithm.
    y_pred = y_pred[:min(len(y_pred), max_compared_results)]

    max_compared_results = min(len(y_true), len(y_pred))
    y_true = y_true[:max_compared_results]
    y_pred = y_pred[:max_compared_results]

    y_true_titles = [Resource.get_comparable_str(r.title) for r in y_true]
    y_pred_titles = [Resource.get_comparable_str(r.title) for r in y_pred]

    for y_pred_title in y_pred_titles:
        if y_pred_title in y_true_titles:
            # Our predicted resource is in the "gold standard" list.
            hit_count += 1
        else:
            # Our predicted resource is not in the "gold standard" list.
            miss_count += 1

    return hit_count / (hit_count + miss_count)


if __name__ == "__main__":
    # Use the IEEE Xplore samples because we need DOI data.
    sample_resources = sr.load_resources_from_json()[
        sr.IEEE_XPLORE_SAMPLE_FILEPATH
    ]
    keyword_model = KeywordRanker.get_model()

    ras: list[float] = []
    for i, resource in enumerate(sample_resources):
        ra = get_recommend_accuracy(resource, keyword_model)
        ras.append(ra)
        print(f"Recommendation accuracy {i}: {ra}")

    mean_ra = sum(ras) / len(ras)
    print(f"Mean recommendation accuracy: {mean_ra}")
