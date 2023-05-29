"""
Objects and methods used to calculate the classification accuracy of the
recommendation algorithm.
"""
import unicodedata
import urllib
import xmltodict
from models.custom_logger import log
from models.evaluation import sample_resources as sr
from models.recommend import get_ranked_resources
from models.resource import Resource
from models.resource_filter import ResourceFilter
from models.resource_rankers.keyword_ranker import KeywordRanker


def get_resource_category_dict(resources):
    """
    :param resources: The resources to collect subject categories for.
    :type resources: list[Resource]
    :return: The mapping between resources and their subject categories.
    :rtype: dict[Resource, list[str]]
    """
    res_cat_dict: dict[Resource, list[str]] = {}

    for resource in resources:
        query_args = {
            "search_query": f"ti:{resource.title.replace(' ', '+')}",
            "start": 0,
            "max_results": 2
        }

        url = "http://export.arxiv.org/api/query?" + "&".join(
            [f"{key}={val}" for key, val in query_args.items()]
        )
        url = unicodedata.normalize("NFKD", url)
        url = url.encode("ascii", "ignore").decode("ascii")
        try:
            res = urllib.request.urlopen(url)
            res = res.read().decode("utf-8")
            res = xmltodict.parse(res)
        except Exception as err:
            log(str(err), "evaluation.classification", "error")
            res_cat_dict[resource] = None
            continue

        log(f"Successful response from {url}", "evaluation.classification")

        if res["feed"]["opensearch:totalResults"]["#text"] == "0":
            res_cat_dict[resource] = None
            continue

        if isinstance(res["feed"]["entry"], list):
            resource_data = res["feed"]["entry"][0]
        else:
            # ArXiv changes the return type if there is only one entry.
            resource_data = res["feed"]["entry"]

        if not isinstance(resource_data, dict):
            log(
                f"Expected type 'dict' from result returned, got '{type(resource_data)}' instead: {resource_data}",
                "evaluation.classification",
                "error"
            )
            res_cat_dict[resource] = None
            continue

        if "category" in resource_data:
            if isinstance(resource_data["category"], list):
                res_cat_dict[resource] = []
                for category_data in resource_data["category"]:
                    res_cat_dict[resource].append(category_data["@term"])
            else:
                res_cat_dict[resource] = [resource_data["category"]["@term"]]
            continue

    return res_cat_dict


def get_classif_precision(target_resource, macro=True):
    """
    Macro-precision considers candidate resources that match the target's wider
    subject area as being a correct classification.
    Micro-precision only considers candidate resources that match the target's
    exact specific subject area as being a correct classification.
    For example, the difference between ``cs`` (computer science in general) and
    ``cs.AI`` (artificial intelligence-specific resources).

    :param target_resource: The target sample resource.
    :type target_resource: Resource
    :param macro: Whether to return macro-precision or micro-precision.
    :type macro: bool
    :return: The classification precision of the recommendation algorithm.
    :rtype: float
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    ranked_resources = get_ranked_resources(
        target_resources=[target_resource],
        existing_resources=[],
        resource_filter=ResourceFilter({}),
        resource_database_ids=[],
        keyword_model=KeywordRanker.get_model()
    )["ranked_database_resources"]

    res_cat_dict = get_resource_category_dict(
        ranked_resources + [target_resource]
    )

    if macro:
        for resource, categories in res_cat_dict.items():
            # For example, extract "cs" from the "cs.AI" category string.
            macro_categories = [c.split(".")[0] for c in categories]
            res_cat_dict[resource] = macro_categories

    for ranked_resource in ranked_resources:
        if res_cat_dict[ranked_resource] is None:
            continue
        if res_cat_dict[target_resource] is None:
            continue

        common_cats = list(set(res_cat_dict[ranked_resource]).intersection(
            res_cat_dict[target_resource]
        ))
        if len(common_cats) > 0:
            # The recommended resource belongs in the same subject category.
            true_positive += 1
        else:
            # The recommended resource belongs in a different subject category.
            false_positive += 1

    return true_positive / (true_positive + false_positive)


if __name__ == "__main__":
    # TODO: Implement.
    pass
