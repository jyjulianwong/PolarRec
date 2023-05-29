"""
Objects and methods used to calculate the recommendation accuracy of the
recommendation algorithm.
"""
import requests
from config import Config
from models.custom_logger import log, log_extended_line
from models.persistent_cache import PersistentCache
from models.evaluation import sample_resources as sr
from models.recommend import get_ranked_resources
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


def get_recommend_precision(target_resource, keyword_model):
    """
    :param target_resource: The target resource.
    :type target_resource: Resource
    :param keyword_model: The word embedding model to be used for keywords.
    :type keyword_model: Word2Vec.KeyedVectors
    :return: The recommendation precision of the recommendation algorithm.
    :rtype: float
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    y_true = S2rAdapter.get_recommended_resources(
        target_resource=target_resource,
        max_results_retd=20
    )
    y_pred = get_ranked_resources(
        target_resources=[target_resource],
        existing_resources=[],
        resource_filter=ResourceFilter({}),
        resource_database_ids=[],
        keyword_model=keyword_model
    )["ranked_database_resources"]
    # Only consider the top 10 resources returned by our algorithm.
    y_pred = y_pred[:min(len(y_pred), 10)]

    max_ress_compared = min(len(y_true), len(y_pred))
    y_true = y_true[:max_ress_compared]
    y_pred = y_pred[:max_ress_compared]

    for pred_res in y_pred:
        if pred_res in y_true:
            # Our predicted resource is in the "gold standard" list.
            true_positive += 1
        else:
            # Our predicted resource is not in the "gold standard" list.
            false_positive += 1

    return true_positive / (true_positive + false_positive)


if __name__ == "__main__":
    sample_resources = sr.load_resources_from_json()[sr.ARXIV_SAMPLE_FILEPATH]
    keyword_model = KeywordRanker.get_model()

    rps: list[float] = []
    for i, resource in enumerate(sample_resources):
        rp = get_recommend_precision(resource, keyword_model)
        rps.append(rp)
        print(f"Recommendation precision {i}: {rp}")

    mean_rp = sum(rps) / len(rps)
    print(f"Mean recommendation precision: {mean_rp}")
